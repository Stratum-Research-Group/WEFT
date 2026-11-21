import pytest
import torch
from weft.kernels.subbit_vlm_triton import fused_subbit_vision_projector

def test_fused_subbit_vision_projector():
    M = 16    # 16 vision patch tokens
    K = 128   # 128 vision feature dimension (4 words of 32 bits)
    N = 64    # 64 language dimension
    
    x = torch.randn(M, K)
    packed_bitboard = torch.randint(-2147483648, 2147483647, (K // 32, N), dtype=torch.int32)
    g_scale = torch.rand(K) + 0.1
    h_scale = torch.rand(N) + 0.1

    out = fused_subbit_vision_projector(x, packed_bitboard, g_scale, h_scale)
    assert out.shape == (M, N)
    assert not torch.isnan(out).any()
    assert torch.norm(out) > 0.0
