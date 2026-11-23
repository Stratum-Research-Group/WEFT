"""
Unit tests for Multi-Head Latent Attention (MLA) in WEFT.
"""

import pytest
import torch
from weft import WeftConfig, WeftForCausalLM


def test_weft_mla_forward_and_decode():
    config = WeftConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_state_heads=2,
        state_rank=8,
        window_size=64,
        use_mla=True,
        mla_latent_dim=64,
    )
    model = WeftForCausalLM(config)
    model.eval()

    # 1. Prefill
    input_ids = torch.randint(0, 1000, (2, 32))
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    assert out.logits.shape == (2, 32, 1000)
    past = out.past_key_values

    # 2. Sequential Decode
    next_tok = torch.tensor([[42], [99]])
    with torch.no_grad():
        out_next = model(next_tok, past_key_values=past, use_cache=True)
    assert out_next.logits.shape == (2, 1, 1000)


def test_weft_mla_gradient_flow():
    config = WeftConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_state_heads=2,
        state_rank=8,
        window_size=64,
        use_mla=True,
        mla_latent_dim=64,
    )
    model = WeftForCausalLM(config)
    model.train()

    input_ids = torch.randint(0, 1000, (2, 16))
    targets = torch.randint(0, 1000, (2, 16))

    out = model(input_ids)
    loss = torch.nn.functional.cross_entropy(out.logits.view(-1, 1000), targets.view(-1))
    loss.backward()

    # Check gradients on MLA projection weights
    layer0 = model.model.layers[0].self_attn
    assert layer0.kv_down_proj.weight.grad is not None
    assert layer0.k_up_proj.weight.grad is not None
    assert layer0.v_up_proj.weight.grad is not None
    assert not torch.isnan(layer0.kv_down_proj.weight.grad).any()
