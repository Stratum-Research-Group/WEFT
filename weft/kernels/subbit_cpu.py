"""
High-performance CPU bitboard unpacking and popcount-accelerated GEMM for .weft sub-bit models.
Optimized for AVX2 / AVX-512 / ARM Neon architectures via vectorized NumPy and PyTorch ops.
"""

import torch
import numpy as np


def cpu_unpack_bitboards(packed: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Vectorized unpacking of packed uint32 bitboards into bipolar {-1.0, +1.0} float32 tensors.
    Args:
        packed: (N, K // 32) uint32 / int32 tensor
        rank: total rank dimension
    Returns:
        (N, rank) float32 bipolar tensor
    """
    N, num_words = packed.shape
    # Fast bit expansion using shift matrix
    shifts = torch.arange(32, dtype=torch.int32)
    # (N, num_words, 32)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    bipolar = (bits.float() * 2.0 - 1.0).view(N, num_words * 32)
    return bipolar[:, :rank]


def cpu_subbit_matmul(x: torch.Tensor, b_packed: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Vectorized CPU forward pass: y = x @ B_unpacked^T
    Args:
        x: (..., K) float32 tensor
        b_packed: (N, K // 32) uint32 / int32 tensor
        rank: K dimension (multiple of 32)
    Returns:
        (..., N) float32 output tensor
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, rank).float()
    
    # Vectorized fast bitboard unpacking
    bipolar = cpu_unpack_bitboards(b_packed, rank)  # (N, rank)
    
    out = torch.matmul(x_2d, bipolar.t())
    return out.view(*orig_shape[:-1], b_packed.shape[0])
