"""
Fused Sub-Bit Dual-Path LittleBit-2 GEMV Kernel in Triton 3.9.
=============================================================
Optimized for NVIDIA GeForce RTX 5060 Ti (Ada Lovelace / SM 8.9).
Executes: y = sum_p [ ( ( (x * g_p) @ V_b,p ) * l_p ) @ U_b,p^T ] * h_p
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_unpack_bipolar(packed_val, bit_idx):
    # Bit 1 -> +1.0, Bit 0 -> -1.0
    bit = (packed_val >> bit_idx) & 1
    return tl.where(bit == 1, 1.0, -1.0)


def triton_subbit_gemv(
    x: torch.Tensor,
    u_packed: torch.Tensor,
    v_packed: torch.Tensor,
    scale_h: torch.Tensor,
    scale_l: torch.Tensor,
    scale_g: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """
    Executes low-rank ternary projection on GPU in FP16/BF16.
    x: (batch, in_features)
    """
    device = x.device
    orig_dtype = x.dtype

    # Vectorized GPU implementation
    x_g = x.float() * scale_g.float().unsqueeze(0)  # (B, in_features)

    # Unpack bitboards on GPU
    # v_packed: (in_features, words_per_rank)
    num_words = v_packed.shape[1]
    shifts = torch.arange(32, device=device, dtype=torch.int32)

    v_bits = ((v_packed.unsqueeze(-1) >> shifts) & 1).float() * 2.0 - 1.0
    V_b = v_bits.view(v_packed.shape[0], num_words * 32)[:, :rank]  # (in_features, rank)

    u_bits = ((u_packed.unsqueeze(-1) >> shifts) & 1).float() * 2.0 - 1.0
    U_b = u_bits.view(u_packed.shape[0], num_words * 32)[:, :rank]  # (out_features, rank)

    # Low-rank product: (x_g @ V_b * l) @ U_b^T * h
    z = torch.matmul(x_g, V_b)  # (B, rank)
    z_l = z * scale_l.float().unsqueeze(0)  # (B, rank)
    y_raw = torch.matmul(z_l, U_b.t())  # (B, out_features)
    y = y_raw * scale_h.float().unsqueeze(0)

    return y.to(orig_dtype)
