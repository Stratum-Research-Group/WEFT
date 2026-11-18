"""
Unit tests for WEFT INT8 and FP8 Quantization subsystems.
"""

import pytest
import torch
from weft import WeftConfig, WeftForCausalLM
from weft.quantization import (
    quantize_weft_int8,
    quantize_weft_fp8,
    get_model_size_mb,
    Int8WeightOnlyLinear,
    FP8Linear,
)


def test_weft_int8_quantization_inference():
    cfg = WeftConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_state_heads=2,
        state_rank=8,
    )
    model = WeftForCausalLM(cfg).to(torch.bfloat16)
    orig_size = get_model_size_mb(model)

    quantize_weft_int8(model)
    quant_size = get_model_size_mb(model)
    assert quant_size < orig_size

    # Verify that decoder self_attn and mlp layers contain Int8WeightOnlyLinear
    assert isinstance(model.model.layers[0].self_attn.q_proj, Int8WeightOnlyLinear)
    assert isinstance(model.model.layers[0].mlp.gate_proj, Int8WeightOnlyLinear)

    # Verify prefill and step decode
    x = torch.randint(0, 1000, (2, 8))
    out = model(x, use_cache=True)
    assert out.logits.shape == (2, 8, 1000)

    step = model(torch.tensor([[10], [20]]), past_key_values=out.past_key_values, use_cache=True)
    assert step.logits.shape == (2, 1, 1000)


def test_weft_fp8_quantization_inference():
    cfg = WeftConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_state_heads=2,
        state_rank=8,
    )
    model = WeftForCausalLM(cfg).to(torch.bfloat16)

    quantize_weft_fp8(model)
    assert isinstance(model.model.layers[0].self_attn.q_proj, FP8Linear)
    assert isinstance(model.model.layers[0].mlp.gate_proj, FP8Linear)

    # Verify prefill and step decode
    x = torch.randint(0, 1000, (2, 8))
    out = model(x, use_cache=True)
    assert out.logits.shape == (2, 8, 1000)

    step = model(torch.tensor([[10], [20]]), past_key_values=out.past_key_values, use_cache=True)
    assert step.logits.shape == (2, 1, 1000)
