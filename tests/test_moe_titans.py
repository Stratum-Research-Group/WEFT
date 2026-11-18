"""
Unit tests for WEFT-2: DeepSeekMoE fine-grained routed experts and Titans plastic memory.
"""

import pytest
import torch
from weft import WeftConfig, WeftForCausalLM


def test_weft_moe_forward_and_aux_loss():
    cfg = WeftConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_state_heads=2,
        state_rank=8,
        use_moe=True,
        moe_num_routed_experts=4,
        moe_top_k=2,
    )
    model = WeftForCausalLM(cfg)
    model.train()

    x = torch.randint(0, 1000, (2, 16))
    out = model(x)
    assert out.logits.shape == (2, 16, 1000)

    # Check that MoE router produced auxiliary loss
    mlp0 = model.model.layers[0].mlp
    aux_loss = mlp0.get_aux_loss()
    assert aux_loss > 0.0

    loss = out.logits.sum() + aux_loss
    loss.backward()

    # Verify router gradients
    assert mlp0.router.weight.grad is not None


def test_weft_titans_forward_and_decode():
    cfg = WeftConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_state_heads=2,
        state_rank=8,
        use_titans=True,
        titans_mem_dim=32,
    )
    model = WeftForCausalLM(cfg)
    model.eval()

    x = torch.randint(0, 1000, (2, 16))
    with torch.no_grad():
        out = model(x, use_cache=True)
    assert out.logits.shape == (2, 16, 1000)

    # Step decode
    tok = torch.tensor([[10], [20]])
    with torch.no_grad():
        out_step = model(tok, past_key_values=out.past_key_values, use_cache=True)
    assert out_step.logits.shape == (2, 1, 1000)
