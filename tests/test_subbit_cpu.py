"""
Unit tests for CPU sub-bit bitboard unpacking and GEMM fallback.
"""

import torch
import pytest
from weft.kernels.subbit_cpu import cpu_unpack_bitboards, cpu_subbit_matmul


def test_cpu_unpack_bitboards():
    N, rank = 16, 64
    words = rank // 32
    packed = torch.randint(0, 0x7FFFFFFF, (N, words), dtype=torch.int32)
    
    unpacked = cpu_unpack_bitboards(packed, rank)
    assert unpacked.shape == (N, rank)
    assert ((unpacked == 1.0) | (unpacked == -1.0)).all(), "Every element must be +1.0 or -1.0"


def test_cpu_subbit_matmul_numerical_parity():
    batch, seq, d_in, d_out, rank = 2, 4, 64, 128, 64
    words = rank // 32
    
    x = torch.randn(batch, seq, rank)
    b_packed = torch.randint(0, 0x7FFFFFFF, (d_out, words), dtype=torch.int32)
    
    # 1. CPU kernel forward
    out = cpu_subbit_matmul(x, b_packed, rank)
    assert out.shape == (batch, seq, d_out)
    
    # 2. Reference manual forward
    b_ref = cpu_unpack_bitboards(b_packed, rank)
    ref_out = x @ b_ref.t()
    
    assert torch.allclose(out, ref_out, atol=1e-5), "CPU sub-bit GEMM diverges from reference"
