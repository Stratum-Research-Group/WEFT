#!/usr/bin/env python3
"""
Black-Box Linear System Identification via Empirical Probing & Matrix Sensing.

Recovers low-rank sub-bit factorizations (U, Sigma, V) of an unknown linear operator
W in R^{m x n} from an embedding/vector endpoint using randomized Gaussian probes:
  Y = W * X
Without ever directly accessing or reading the underlying model weights.
"""

import argparse
import time
import torch
import numpy as np
from typing import Callable, Tuple

from weft.converter import factorize_linear_layer


def probe_randomized_svd_linear_operator(
    probe_fn: Callable[[torch.Tensor], torch.Tensor],
    in_features: int,
    target_rank: int = 64,
    oversample: int = 10,
    itq_iterations: int = 3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Recovers low-rank factorization of a linear operator W from black-box probes.

    Args:
      probe_fn: Callable mapping [B, in_features] -> [B, out_features]
      in_features: Known or estimated input dimension n
      target_rank: Intrinsic rank r to extract (e.g. 64)
      oversample: Number of extra Gaussian probes p (typically 5-10)
      itq_iterations: Polar Joint-ITQ iterations for sub-bit hypercube alignment

    Returns:
      U_b: Bipolar hypercube bitboard {-1, +1}^{m x r}
      V_b: Bipolar hypercube bitboard {-1, +1}^{n x r}
      scales: Channel-wise float16 scale factors
      elapsed_sec: Total probing time
    """
    total_probes = target_rank + oversample
    t0 = time.time()

    # 1. Generate randomized Gaussian sketch matrix X in R^{n x total_probes}
    # Using float32 for maximum probe stability
    torch.manual_seed(42)
    X = torch.randn(in_features, total_probes, dtype=torch.float32)

    # 2. Query the black-box endpoint: Y = W * X in R^{m x total_probes}
    # Batched forward pass
    with torch.no_grad():
        Y = probe_fn(X.T).T  # [out_features, total_probes]

    out_features = Y.shape[0]

    # 3. Form orthonormal basis Q via QR factorization: Y = Q * R
    Q, _ = torch.linalg.qr(Y, mode="reduced")  # Q in R^{out_features x total_probes}

    # 4. Invert projection on input space:
    # Since Y = W * X, Q.T * Y = (Q.T * W) * X
    # Let B = Q.T * W in R^{total_probes x in_features}
    # B = (Q.T * Y) * X^+ (pseudo-inverse of X)
    X_pinv = torch.linalg.pinv(X)  # [total_probes, in_features]
    B = torch.matmul(torch.matmul(Q.T, Y), X_pinv)  # [total_probes, in_features]

    # 5. SVD of the small matrix B
    U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)

    # Truncate to target_rank
    U_tilde = U_tilde[:, :target_rank]
    S = S[:target_rank]
    V = Vh[:target_rank, :].T  # [in_features, target_rank]

    # Reconstruct operator left singular vectors: U = Q * U_tilde in R^{out_features x target_rank}
    U = torch.matmul(Q, U_tilde)  # [out_features, target_rank]

    # Low-rank reconstructed dense surrogate: W_rec = U * diag(S) * V.T
    W_rec = torch.matmul(U * S.unsqueeze(0), V.T)

    # 6. Apply WEFT Polar Joint-ITQ to map to sub-bit bitboards {-1, +1}
    subbit_layer = factorize_linear_layer(
        W_rec,
        base_rank=target_rank,
        num_paths=1,
        itq_iterations=itq_iterations
    )

    elapsed_sec = time.time() - t0
    return subbit_layer, W_rec, elapsed_sec


def run_synthetic_linear_demonstration():
    """Validates black-box operator probing against a hidden reference ground truth."""
    print("=" * 68)
    print("  BLACK-BOX LINEAR OPERATOR IDENTIFICATION DEMONSTRATION")
    print("=" * 68)

    # Ground truth hidden linear layer (e.g., d_out=4096, d_in=4096, intrinsic rank 64)
    d_out, d_in, true_rank = 1024, 1024, 48
    print(f"Synthesizing hidden operator: W in R^{d_out}x{d_in} (intrinsic rank={true_rank})...")

    U_true = torch.randn(d_out, true_rank)
    V_true = torch.randn(d_in, true_rank)
    W_hidden = torch.matmul(U_true, V_true.T) / (true_rank ** 0.5)

    # Simulated Black-Box API Function (no access to W_hidden internals)
    api_call_count = [0]
    def blackbox_api_endpoint(x: torch.Tensor) -> torch.Tensor:
        api_call_count[0] += x.shape[0]
        # Evaluates strictly y = x @ W.T
        return torch.matmul(x, W_hidden.T)

    target_rank = 64
    print(f"Launching randomized probe sensing (target_rank={target_rank}, oversample=10)...")
    subbit_layer, W_rec, elapsed = probe_randomized_svd_linear_operator(
        probe_fn=blackbox_api_endpoint,
        in_features=d_in,
        target_rank=target_rank,
        oversample=10,
        itq_iterations=3
    )

    # Test reconstruction fidelity on unseen test vectors
    x_test = torch.randn(10, d_in)
    y_true = blackbox_api_endpoint(x_test)
    y_rec = subbit_layer(x_test)

    cos_sim = torch.cosine_similarity(y_true, y_rec, dim=-1).mean().item()
    rel_err = (torch.norm(y_true - y_rec) / torch.norm(y_true)).item()

    print("-" * 68)
    print(f"Total API Probes Injected:     {api_call_count[0]} vectors (1 batched matrix call)")
    print(f"Elapsed Probing & ITQ Time:    {elapsed:.3f} seconds")
    print(f"Sub-Bit Compression Rate:      {subbit_layer.bits_per_weight():.4f} BPW")
    print(f"Unseen Test Cosine Similarity: {cos_sim * 100:.2f}%")
    print(f"Relative Reconstruction Error: {rel_err * 100:.2f}%")
    print("=" * 68)


if __name__ == "__main__":
    run_synthetic_linear_demonstration()
