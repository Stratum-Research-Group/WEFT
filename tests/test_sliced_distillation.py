"""
Unit test for Zero-RAM Sliced Frontier Model Distillation.
Verifies that student layers can align with probed teacher representations
without loading complete frontier models into memory.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest

from weft import WeftConfig, WeftForCausalLM
from scripts.distill_teacher_to_weft import SlicedLayerDistiller, ZeroRamTeacherProber, run_distillation_step


def test_zero_ram_sliced_distillation():
    cfg = WeftConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=16,
    )
    student = WeftForCausalLM(cfg)
    teacher_dim = 256
    distiller = SlicedLayerDistiller(student_dim=cfg.hidden_size, teacher_dim=teacher_dim)
    prober = ZeroRamTeacherProber(teacher_dim=teacher_dim)

    inputs = torch.randint(0, cfg.vocab_size, (2, 16))
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(distiller.parameters()), lr=1e-3)

    initial_loss = run_distillation_step(
        student_model=student,
        distiller=distiller,
        prober=prober,
        input_ids=inputs,
        layer_to_probe=0,
    )["distill_feature_loss"]

    # Run 3 training steps
    for _ in range(3):
        run_distillation_step(
            student_model=student,
            distiller=distiller,
            prober=prober,
            input_ids=inputs,
            layer_to_probe=0,
            optimizer=optimizer,
        )

    eval_loss = run_distillation_step(
        student_model=student,
        distiller=distiller,
        prober=prober,
        input_ids=inputs,
        layer_to_probe=0,
    )["distill_feature_loss"]

    assert eval_loss < initial_loss, f"Expected distillation loss to decrease: {eval_loss} vs {initial_loss}"
