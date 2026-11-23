"""
Unit tests for BitNet b1.58 Ternary Quantization in WEFT-2.
"""

import pytest
import torch
from weft.configuration_weft import WeftConfig
from weft.modeling_weft import WeftForCausalLM
from weft.quantization_b158 import BitLinear158, convert_weft_to_bitnet, weight_quant_b158


def test_bitlinear_ternary_properties():
    layer = BitLinear158(64, 128)
    w_q, scale = weight_quant_b158(layer.weight)
    
    unique_vals = torch.unique(w_q).tolist()
    for v in unique_vals:
        assert v in [-1.0, 0.0, 1.0]

    x = torch.randn(2, 8, 64)
    y = layer(x)
    assert y.shape == (2, 8, 128)
    assert not torch.isnan(y).any()


def test_convert_weft_to_bitnet():
    cfg = WeftConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_state_heads=2,
        state_rank=8,
        window_size=32,
    )
    model = WeftForCausalLM(cfg)
    convert_weft_to_bitnet(model)

    # Confirm BitLinear layers replaced nn.Linear in layers
    assert isinstance(model.model.layers[0].self_attn.q_proj, BitLinear158)
    
    # Test forward pass with ternary model
    x = torch.randint(0, 1000, (1, 16))
    out = model(x)
    assert out.logits.shape == (1, 16, 1000)
