"""
Unit & Integration Test Suite for Cerebro-Cerebellar Forward Internal Model Dynamics
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from weft.kernels.fused_cortico_cerebellar_forward_model_triton import fused_forward_sensory_predictor
from scripts.sim_forward_sensory_cancellation import run_forward_sensory_sim

def test_forward_model_simulation():
    res = run_forward_sensory_sim()
    assert res["status"] == "success"
    assert res["reafference_cancellation_verified"] is True
    assert res["cancellation_ratio"] > 0.85
    assert res["perturbation_peak_discrepancy"] > 0.60
    assert res["unperturbed_residual_mean"] < 0.05

def test_forward_sensory_predictor_kernel():
    B = 4
    D_M = 8
    D_S = 8
    
    # Simple identity forward model matrix
    w_fwd = torch.eye(D_S, D_M, dtype=torch.float32)
    u_motor = torch.ones((B, D_M), dtype=torch.float32)
    # Perfect match sensory outcome
    y_actual = torch.ones((B, D_S), dtype=torch.float32)

    y_pred, e_disc, ratio = fused_forward_sensory_predictor(u_motor, y_actual, w_fwd, cancellation_gain=1.0)

    # Discrepancy should be zero
    assert torch.allclose(e_disc, torch.zeros_like(e_disc), atol=1e-5)
    assert ratio > 0.99

    # Add external perturbation
    y_perturbed = y_actual.clone()
    y_perturbed[:, 0] += 5.0
    _, e_disc_pert, ratio_pert = fused_forward_sensory_predictor(u_motor, y_perturbed, w_fwd, cancellation_gain=1.0)
    
    assert torch.max(torch.abs(e_disc_pert)).item() >= 4.9
    assert ratio_pert < ratio

def test_forward_model_avx2_binary():
    bin_path = Path("build/weft-bench-forward-model")
    assert bin_path.exists()
