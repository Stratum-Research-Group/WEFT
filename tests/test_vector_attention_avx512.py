import math
import torch
import pytest

def reference_causal_attention(q, k, v):
    # q, k, v: [seq_len, head_dim]
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    seq_len = q.shape[0]
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    scores = scores + mask
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)

def test_causal_attention_numerical_stability():
    seq_len = 16
    head_dim = 64
    torch.manual_seed(42)
    q = torch.randn(seq_len, head_dim)
    k = torch.randn(seq_len, head_dim)
    v = torch.randn(seq_len, head_dim)

    out = reference_causal_attention(q, k, v)
    assert out.shape == (seq_len, head_dim)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    print("[PASS] Vector attention reference numerical verification passed.")

if __name__ == "__main__":
    test_causal_attention_numerical_stability()
