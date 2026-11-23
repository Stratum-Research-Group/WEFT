"""
Regression test for BitNet b1.58 Ternary Speculation Benchmark.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from weft import WeftConfig, WeftForCausalLM
from scripts.benchmark_ternary_speculation import load_ternary_weights_into_model
from scripts.quantize_distilled_to_bitnet import quantize_weight_ternary


ckpt_root = os.environ.get("WEFT_CHECKPOINTS_ROOT", "/run/media/akarta/AI/checkpoints")
TERNARY_CKPT = os.path.join(ckpt_root, "deep_distill", "student_ternary_bitweft.pt")


def test_quantize_weight_ternary():
    w = torch.tensor([[-2.5, 0.05, 1.8], [0.0, -1.2, 3.1]])
    t_w, scale = quantize_weight_ternary(w)
    assert t_w.dtype == torch.int8
    assert torch.all((t_w == -1) | (t_w == 0) | (t_w == 1))
    assert scale.item() > 0


@pytest.mark.skipif(not os.path.exists(TERNARY_CKPT), reason="Ternary checkpoint not found on AI drive")
def test_load_ternary_weights():
    cfg = WeftConfig(
        vocab_size=262144,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=7,
        num_attention_heads=8,
        max_position_embeddings=16,
    )
    model = WeftForCausalLM(cfg)
    load_ternary_weights_into_model(model, TERNARY_CKPT)
    x = torch.randint(0, 1000, (1, 4))
    with torch.no_grad():
        out = model(x)
    assert out.logits.shape == (1, 4, 262144)
    assert not torch.isnan(out.logits).any()
