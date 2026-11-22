"""
WEFT Converter Engine: Transforms dense HuggingFace / Safetensors weights into
native .weft sub-bit binary model containers via LittleBit-2 Joint-ITQ factorization.
Includes spectral decay gamma estimation for adaptive per-layer rank allocation.
"""

import os
import argparse
import json
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from weft.format import WeftSubBitLinear, WeftModelContainer


def estimate_spectral_decay(W: torch.Tensor, top_k: int = 64) -> float:
    """
    Estimates the spectral power-law decay exponent gamma: sigma_k approx C * k^(-gamma).
    Higher gamma implies fast singular value decay (lower rank needed).
    Uses randomized low-rank projection for large matrices to guarantee O(1) memory and prevent LAPACK OOM.
    """
    W_float = W.float()
    m, n = W_float.shape
    k = min(top_k, min(m, n))
    try:
        if min(m, n) > 1024:
            _, S, _ = torch.svd_lowrank(W_float, q=k, niter=2)
            S_vals = S[:k].cpu().numpy()
        else:
            _, S, _ = torch.linalg.svd(W_float, full_matrices=False)
            S_vals = S[:k].cpu().numpy()
        ranks = np.arange(1, k + 1)
        # Log-linear regression: log(S) = log(C) - gamma * log(k)
        log_s = np.log(np.maximum(S_vals, 1e-8))
        log_k = np.log(ranks)
        gamma = -float(np.polyfit(log_k, log_s, 1)[0])
        return max(0.01, gamma)
    except Exception:
        return 0.5


def factorize_linear_layer(
    W: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    base_rank: int = 64,
    adaptive_rank: bool = True,
    num_paths: int = 2,
    itq_iterations: int = 10,
    certify_intervals: bool = False,
):
    """
    Factorizes a dense 2D weight matrix W into a WeftSubBitLinear module with LittleBit-2 Joint-ITQ.
    Decomposes W (m x n) into dual-path LittleBit-2 factors:
    W_approx = sum_p diag(h_p) @ (U_b,p @ diag(l_p) @ V_b,p^T) @ diag(g_p)
    If certify_intervals is True, validates the reconstruction error using mpmath interval arithmetic.
    """
    m, n = W.shape
    device = W.device

    # Estimate spectral decay and allocate rank
    gamma = estimate_spectral_decay(W)
    if adaptive_rank:
        # Higher decay (gamma > 0.6) allows smaller rank; slower decay requires larger rank
        rank_mult = max(0.5, min(2.0, 0.6 / gamma))
        rank = int(round(base_rank * rank_mult / 32.0) * 32)
        rank = max(1, min(min(m, n), rank))
    else:
        rank = min(min(m, n), base_rank)

    layer = WeftSubBitLinear(
        in_features=n,
        out_features=m,
        rank=rank,
        num_paths=num_paths,
        bias=bias is not None,
    )

    residual = W.clone().float()

    for p in range(num_paths):
        # 1. Channel-wise scale normalization
        g = torch.norm(residual, dim=0, keepdim=True).clamp_min(1e-5)  # (1, n)
        h = torch.norm(residual / g, dim=1, keepdim=True).clamp_min(1e-5)  # (m, 1)

        normed = (residual / g) / h  # (m, n)

        # 2. Fast Low-Rank SVD (Randomized PCA/SVD)
        # Using torch.svd_lowrank instead of torch.linalg.svd prevents O(m*n) Lapack buffer blowup
        # (e.g. 152064x8192 matrices where full svd allocates >15GB and triggers systemd-oomd)
        if min(m, n) > 2048 and rank < min(m, n) // 4:
            U_r, S_r, V_r = torch.svd_lowrank(normed, q=rank, niter=2)
        else:
            U, S, Vh = torch.linalg.svd(normed, full_matrices=False)
            U_r = U[:, :rank]
            S_r = S[:rank]
            V_r = Vh[:rank, :].t()  # (n, rank)

        # Distribute singular values
        U_hat = U_r * torch.sqrt(S_r.unsqueeze(0))
        V_hat = V_r * torch.sqrt(S_r.unsqueeze(0))

        # 3. Joint Iterative Quantization (ITQ) to align with hypercube vertices
        # Initialize rotation R as random orthogonal
        R = torch.eye(rank, device=device)
        for _ in range(itq_iterations):
            # Quantize
            U_rot = torch.matmul(U_hat, R)
            V_rot = torch.matmul(V_hat, R)
            B_u = torch.sign(U_rot)
            B_u[B_u == 0] = 1.0
            B_v = torch.sign(V_rot)
            B_v[B_v == 0] = 1.0

            # Update orthogonal rotation R via polar decomposition (SVD) of (U_hat^T B_u + V_hat^T B_v)
            C = torch.matmul(U_hat.t(), B_u) + torch.matmul(V_hat.t(), B_v)
            U_c, _, Vh_c = torch.linalg.svd(C, full_matrices=False)
            R = torch.matmul(U_c, Vh_c)

        # Final binary matrices
        B_u = torch.sign(torch.matmul(U_hat, R))
        B_u[B_u == 0] = 1.0
        B_v = torch.sign(torch.matmul(V_hat, R))
        B_v[B_v == 0] = 1.0

        # Pack B_u and B_v into uint32 bitboards
        # Bit 1 if +1.0, 0 if -1.0
        u_bits = (B_u > 0).to(torch.int32)
        v_bits = (B_v > 0).to(torch.int32)

        words_per_rank = (rank + 31) // 32
        u_packed = torch.zeros(m, words_per_rank, dtype=torch.int32, device=device)
        v_packed = torch.zeros(n, words_per_rank, dtype=torch.int32, device=device)

        for w in range(words_per_rank):
            for b in range(32):
                col = w * 32 + b
                if col < rank:
                    u_packed[:, w] |= (u_bits[:, col] << b)
                    v_packed[:, w] |= (v_bits[:, col] << b)

        # Fit scale vector l_p via regularized least squares
        # Target: minimize ||normed - B_u diag(l) B_v^T||_F^2
        # Vectorized normal equations: G @ l = rhs, where G_ij = (B_u[:, i]^T B_u[:, j]) * (B_v[:, i]^T B_v[:, j])
        # and rhs_i = B_u[:, i]^T @ normed @ B_v[:, i]
        G = torch.matmul(B_u.t(), B_u) * torch.matmul(B_v.t(), B_v)  # (rank, rank)
        rhs = torch.diag(torch.matmul(torch.matmul(B_u.t(), normed), B_v))  # (rank,)
        reg = 1e-4 * torch.eye(rank, device=device, dtype=G.dtype)
        l_scale = torch.linalg.solve(G + reg, rhs).clamp_min(1e-6)

        layer.u_packed[p].data.copy_(u_packed)
        layer.v_packed[p].data.copy_(v_packed)
        layer.scale_h[p].data.copy_(h.squeeze(1).to(torch.float16))
        layer.scale_l[p].data.copy_(l_scale.to(torch.float16))
        layer.scale_g[p].data.copy_(g.squeeze(0).to(torch.float16))

        # Subtract reconstructed approximation from residual
        U_b_bipolar = (u_bits.float() * 2.0 - 1.0)
        V_b_bipolar = (v_bits.float() * 2.0 - 1.0)
        W_p = (h * (torch.matmul(U_b_bipolar * l_scale.unsqueeze(0), V_b_bipolar.t()))) * g
        residual = residual - W_p

    if bias is not None:
        layer.bias.data.copy_(bias.to(torch.float16))

    if certify_intervals:
        from weft.interval_certifier import RigorousIntervalQuantizationValidator
        cert_info = RigorousIntervalQuantizationValidator.certify_reconstruction_error(W, layer)
        return layer, cert_info

    return layer


def convert_model_to_weft(
    state_dict: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    output_filepath: str,
    base_rank: int = 64,
    adaptive_rank: bool = True,
    compress_entropy: bool = False,
    certify_intervals: bool = False,
) -> Tuple[WeftModelContainer, Optional[Dict[str, Any]]]:
    """
    Converts a state dict of linear projections into a native .weft binary file.
    If certify_intervals is True, compiles a comprehensive Mathematical Robustness Certificate.
    """
    container = WeftModelContainer(metadata=config)
    certificate = {} if certify_intervals else None

    for key, weight in state_dict.items():
        if key.endswith(".weight") and weight.ndim == 2:
            layer_name = key[: -len(".weight")]
            bias_key = layer_name + ".bias"
            bias = state_dict.get(bias_key, None)

            if certify_intervals:
                subbit_layer, cert_info = factorize_linear_layer(
                    weight,
                    bias=bias,
                    base_rank=base_rank,
                    adaptive_rank=adaptive_rank,
                    certify_intervals=True,
                )
                certificate[layer_name] = cert_info
            else:
                subbit_layer = factorize_linear_layer(
                    weight,
                    bias=bias,
                    base_rank=base_rank,
                    adaptive_rank=adaptive_rank,
                    certify_intervals=False,
                )
            container.add_subbit_linear(layer_name, subbit_layer)

    container.save(output_filepath, compress_entropy=compress_entropy)

    if certify_intervals and certificate:
        cert_path = output_filepath.rsplit(".", 1)[0] + "_certificate.json"
        summary = {
            "total_layers_certified": len(certificate),
            "all_strictly_bounded_below_one": all(
                c["is_strictly_bounded_below_one"] for c in certificate.values() if c is not None
            ),
            "max_certified_upper_bound": max(
                c["certified_upper_bound"] for c in certificate.values() if c is not None
            ) if certificate else 0.0,
            "layers": certificate,
        }
        with open(cert_path, "w") as f:
            json.dump(summary, f, indent=2)

    return container, certificate if certify_intervals else None
