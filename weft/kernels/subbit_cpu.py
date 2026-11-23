"""
High-performance CPU bitboard unpacking and popcount-accelerated GEMM for .weft sub-bit models.
Optimized for AVX-512 VPOPCNTDQ (Zen 4 / Sapphire Rapids) with fallback to PyTorch vectorization.
"""

import torch
import numpy as np

# Dynamically import native AVX-512 hardware accelerator if available on host
_AVX512_AVAILABLE = False
try:
    import weft_avx512
    _AVX512_AVAILABLE = True
except ImportError:
    try:
        from ... import weft_avx512
        _AVX512_AVAILABLE = True
    except Exception:
        _AVX512_AVAILABLE = False


def is_avx512_accelerated() -> bool:
    """Returns True if native AVX-512 VPOPCNTDQ acceleration is active."""
    return _AVX512_AVAILABLE


def cpu_unpack_bitboards(packed: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Vectorized unpacking of packed uint32/uint64 bitboards into bipolar {-1.0, +1.0} float32 tensors.
    """
    N, num_words = packed.shape
    shifts = torch.arange(32, dtype=torch.int32, device=packed.device)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    bipolar = (bits.float() * 2.0 - 1.0).view(N, num_words * 32)
    return bipolar[:, :rank]


def cpu_subbit_matmul(x: torch.Tensor, b_packed: torch.Tensor, rank: int, scale: float = 1.0) -> torch.Tensor:
    """
    Vectorized CPU forward pass: y = x @ B_unpacked^T
    Automatically routes through native AVX-512 hardware popcount when available.
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, rank)
    batch_size = x_2d.shape[0]

    # Fast Path: AVX-512 Hardware VPOPCNTDQ
    if _AVX512_AVAILABLE and b_packed.dtype == torch.int64:
        cols_u64 = rank // 64
        # Quantize activations to signs
        x_sign = (x_2d > 0).to(torch.int64)

        if batch_size == 4 and hasattr(weft_avx512, "subbit_gemm_avx512_batch4"):
            # Batch-4 GEMM with register reuse
            out = torch.zeros(4, b_packed.shape[0], dtype=torch.float32, device=x.device)
            weft_avx512.subbit_gemm_avx512_batch4(b_packed, x_sign, out, scale)
            return out.view(*orig_shape[:-1], b_packed.shape[0])
        elif batch_size == 1 and hasattr(weft_avx512, "subbit_gemv_avx512"):
            # Single-token GEMV
            out = torch.zeros(b_packed.shape[0], dtype=torch.float32, device=x.device)
            weft_avx512.subbit_gemv_avx512(b_packed, x_sign.squeeze(0), out, scale)
            return out.view(*orig_shape[:-1], b_packed.shape[0])

    # Fallback Path: Standard Vectorized Unpacking
    bipolar = cpu_unpack_bitboards(b_packed, rank)
    out = torch.matmul(x_2d.float(), bipolar.t()) * scale
    return out.view(*orig_shape[:-1], b_packed.shape[0])
