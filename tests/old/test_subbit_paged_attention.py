"""
Test suite for Fused Sub-Bit PagedAttention Kernel.
"""

import torch
import pytest
from weft.kernels.subbit_paged_attention import SubBitPagedAttention


def test_subbit_paged_attention_forward():
    b, s, d = 2, 16, 256
    heads, head_dim, rank = 4, 64, 32
    
    attn = SubBitPagedAttention(hidden_size=d, num_heads=heads, head_dim=head_dim, rank=rank)
    x = torch.randn(b, s, d)
    
    out = attn(x)
    assert out.shape == (b, s, d)
    assert not torch.isnan(out).any()


def test_subbit_paged_attention_causal():
    b, s, d = 1, 8, 128
    attn = SubBitPagedAttention(hidden_size=d, num_heads=2, head_dim=64, rank=16)
    
    x1 = torch.randn(b, s, d)
    x2 = x1.clone()
    # Mutate future token
    x2[:, -1, :] += 5.0
    
    out1 = attn(x1)
    out2 = attn(x2)
    
    # First token output must not depend on future tokens due to causal masking
    assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-4)
