"""
Unit tests for BitNet b1.58 Addition-Only Triton GEMM Kernel.
"""

import pytest
import torch
from weft.kernels.ternary_gemm import (
    pack_ternary_weights,
    unpack_ternary_weights,
    ternary_gemm_triton,
)
from weft.quantization_b158 import act_quant_int8


def test_ternary_packing_roundtrip():
    out_f, in_f = 64, 128
    w = torch.randn(out_f, in_f)
    packed, gamma = pack_ternary_weights(w)
    unpacked = unpack_ternary_weights(packed, out_f, in_f)

    # Reference ternary values
    w_ref = torch.round(w / gamma).clamp(-1.0, 1.0).to(torch.int8)
    diff = (unpacked - w_ref).abs().max().item()
    assert diff == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel")
def test_ternary_gemm_triton_parity():
    device = "cuda"
    M, K, N = 32, 256, 64
    x = torch.randn(M, K, device=device)
    w = torch.randn(N, K, device=device)

    x_int8, scale_x = act_quant_int8(x)
    w_packed, gamma = pack_ternary_weights(w)

    y_triton = ternary_gemm_triton(x_int8.to(torch.int8), scale_x, w_packed, gamma, dtype=torch.float32)

    w_unpacked = unpack_ternary_weights(w_packed, N, K).float()
    y_ref = (x_int8.float() @ w_unpacked.T) * (gamma.T / scale_x)

    cos_sim = torch.cosine_similarity(y_triton.flatten(), y_ref.flatten(), dim=0).item()
    assert cos_sim > 0.9999
