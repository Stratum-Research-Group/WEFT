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
    past = out.past_key_values
    w_before = past.layers[0].weft_titans_state[0].clone()
    m_before = past.layers[0].weft_titans_state[1].clone()

    with torch.no_grad():
        out_step = model(tok, past_key_values=past, use_cache=True)
    assert out_step.logits.shape == (2, 1, 1000)

    # Verify Titans state actually advances during decode
    w_after = out_step.past_key_values.layers[0].weft_titans_state[0]
    m_after = out_step.past_key_values.layers[0].weft_titans_state[1]
    assert not torch.equal(w_before, w_after), "Titans W fast weight must update during decode"
    assert not torch.equal(m_before, m_after), "Titans m momentum state must update during decode"


def test_titans_same_schedule_exact_parity():
    """
    Validation Gate: When chunk_size=1 (identical schedule), full-sequence Titans
    forward and token-by-token decode must match to floating-point machine precision.
    """
    from weft.titans import TitansSurpriseMemory

    titans = TitansSurpriseMemory(d_model=32, d_mem=16, chunk_size=1)
    titans.eval()

    torch.manual_seed(42)
    x = torch.randn(2, 6, 32)

    with torch.no_grad():
        out_seq, (W_seq, m_seq) = titans(x)

    with torch.no_grad():
        state = None
        step_outs = []
        for t in range(6):
            xt = x[:, t : t + 1, :]
            ot, state = titans(xt, fast_state=state)
            step_outs.append(ot)
        out_steps = torch.cat(step_outs, dim=1)
        W_steps, m_steps = state

    assert (out_seq - out_steps).abs().max().item() < 1e-6
    assert (W_seq - W_steps).abs().max().item() < 1e-6
    assert (m_seq - m_steps).abs().max().item() < 1e-6
