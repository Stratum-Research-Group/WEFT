"""
Test suite for Scale Residual Distillation Tuning.
"""

import torch
import pytest
from weft.format import WeftSubBitLinear
from weft.scale_tuner import tune_scale_residuals
from weft.converter import factorize_linear_layer


def test_scale_residual_distillation_improvement():
    d_out, d_in, rank = 128, 64, 32
    W = torch.randn(d_out, d_in)

    # Initial factorization
    subbit = factorize_linear_layer(W, base_rank=rank, adaptive_rank=False, num_paths=2)

    # Initial error
    x = torch.randn(4, 16, d_in)
    with torch.no_grad():
        y_ref = x @ W.t()
        y_init = subbit(x).float()
        init_err = (torch.norm(y_init - y_ref) / torch.norm(y_ref)).item()

    # Distillation tuning on scale parameters
    final_err = tune_scale_residuals(subbit, W, num_steps=30, lr=5e-2)

    assert final_err < init_err, f"Distillation did not reduce reconstruction error: {final_err:.4f} >= {init_err:.4f}"
