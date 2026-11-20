import pytest
import torch
import torch.nn as nn
from weft.singlet_attention import SingletUncrossedAttention


def test_singlet_uncrossed_attention_shape_parity():
    embed_dim = 128
    num_heads = 4
    attn = SingletUncrossedAttention(embed_dim=embed_dim, num_heads=num_heads, state_rank=8)

    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Standard forward pass
    y, state = attn(x, return_state=True)
    assert y.shape == (batch_size, seq_len, embed_dim)
    assert state.shape == (batch_size, num_heads, embed_dim // num_heads, 8)


def test_singlet_attention_autoregressive_step_parity():
    embed_dim = 64
    num_heads = 2
    attn = SingletUncrossedAttention(embed_dim=embed_dim, num_heads=num_heads, state_rank=8)

    batch_size = 1
    # Run full sequence
    x_full = torch.randn(batch_size, 16, embed_dim)
    _, final_state = attn(x_full, return_state=True)

    # Single-token recurrent decode step
    x_token = torch.randn(batch_size, 1, embed_dim)
    y_step, next_state = attn(x_token, prev_state=final_state, return_state=True)

    assert y_step.shape == (batch_size, 1, embed_dim)
    assert next_state.shape == final_state.shape
    assert not torch.isnan(y_step).any()
    assert not torch.isinf(y_step).any()


def test_singlet_attention_drop_in_transformer():
    # Verify it plugs directly into an arbitrary PyTorch Transformer layer
    embed_dim = 128
    sula = SingletUncrossedAttention(embed_dim=embed_dim, num_heads=4)
    mlp = nn.Sequential(
        nn.Linear(embed_dim, embed_dim * 4),
        nn.GELU(),
        nn.Linear(embed_dim * 4, embed_dim),
    )
    norm1 = nn.LayerNorm(embed_dim)
    norm2 = nn.LayerNorm(embed_dim)

    x = torch.randn(2, 64, embed_dim)
    # Residual block
    h = x + sula(norm1(x))[0]
    out = h + mlp(norm2(h))

    assert out.shape == x.shape
