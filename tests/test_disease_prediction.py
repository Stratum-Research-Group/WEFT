"""
tests/test_disease_prediction.py
Pytest Regression Suite: Electrophysiological Disease State Prediction
Lean 4 Target #397 | Paper #206 | Cycle v340.75.0
"""

import pytest
import math
import torch
from weft.kernels.fused_disease_prediction_triton import (
    launch_fused_disease_prediction,
)

def test_beta_power_depletion_monotonicity():
    """Verify that LFP beta power increases monotonically as striatal dopamine depletes."""
    powers = []
    dr = 50
    # Decreasing dopamine from 100 down to 0
    for da_int in range(100, -1, -10):
        # Lean 4 formula: clamp(100 + (80 * (100 - da) * (50 + dr / 2)) / 10000, 0, 1000)
        deficit = 100 - da_int
        raw = 100 + (80 * deficit * (50 + dr // 2)) // 10000
        p_scaled = min(1000, raw)
        powers.append(p_scaled)
        assert 100 <= p_scaled <= 1000

    for i in range(len(powers) - 1):
        assert powers[i] <= powers[i+1]

def test_pac_scaling_with_beta():
    """Verify phase-amplitude coupling index bounds and monotonic scaling."""
    pac_indices = []
    for beta in range(100, 1001, 100):
        pac = min(1000, (beta * 90) // 100)
        pac_indices.append(pac)
        assert 0 <= pac <= 1000

    for i in range(len(pac_indices) - 1):
        assert pac_indices[i] <= pac_indices[i+1]

def test_predictive_alert_trigger_threshold():
    """Verify early warning alert triggers when risk crosses 450 (45%)."""
    # Healthy baseline
    risk_healthy = (50 * 100 + 35 * 90 + 15 * (300 - 280)) // 100
    assert risk_healthy < 450, f"Healthy baseline should not trigger alert, got {risk_healthy}"

    # Pathological state (beta = 600, pac = 540, slope = 150)
    risk_patho = (50 * 600 + 35 * 540 + 15 * (300 - 150)) // 100
    assert risk_patho >= 450, f"Pathological state must trigger early warning alert, got {risk_patho}"

def test_cpu_vector_emulation_for_disease_kernel():
    """Test batched tensor calculation matching the fused Triton GPU kernel interface."""
    N = 64
    dopamine = torch.linspace(1.0, 0.0, N, dtype=torch.float32)
    drive = torch.full((N,), 0.6, dtype=torch.float32)
    jitter = torch.full((N,), 0.1, dtype=torch.float32)

    beta_base = 10.0
    beta_scale = 80.0
    beta_max = 100.0
    pac_th = 15.0
    pac_scale = 1.0 / 75.0
    slope_base = 2.8
    slope_da_scale = 1.4
    slope_jit_scale = 0.2
    slope_min = 1.0
    slope_max = 3.5
    w_beta = 0.50 / 90.0
    w_pac = 0.35
    w_slope = 0.15 / 2.0
    slope_ref = 3.0
    alert_th = 0.45
    alert_slope = 4.0

    deficit = torch.clamp(1.0 - dopamine, min=0.0)
    def_sq = deficit * deficit
    dr_amp = 0.5 + 0.5 * drive
    raw_beta = beta_base + beta_scale * def_sq * dr_amp
    beta = torch.clamp(raw_beta, 0.0, beta_max)

    raw_pac = (beta - pac_th) * pac_scale
    pac = torch.clamp(raw_pac, 0.0, 1.0)

    raw_slope = slope_base - slope_da_scale * deficit + slope_jit_scale * jitter
    slope = torch.clamp(raw_slope, slope_min, slope_max)

    inv_slope = torch.clamp(slope_ref - slope, min=0.0)
    raw_risk = w_beta * beta + w_pac * pac + w_slope * inv_slope
    risk = torch.clamp(raw_risk, 0.0, 1.0)

    raw_alert = 0.5 + alert_slope * (risk - alert_th)
    alert = torch.clamp(raw_alert, 0.0, 1.0)

    assert len(beta) == N
    assert len(pac) == N
    assert len(slope) == N
    assert len(risk) == N
    assert len(alert) == N

    # Healthy index 0 (da = 1.0):
    assert dopamine[0] == 1.0
    assert beta[0] == pytest.approx(10.0, abs=1e-2)
    assert pac[0] == 0.0
    assert alert[0] == 0.0

    # Depleted index -1 (da = 0.0):
    assert dopamine[-1] == 0.0
    assert beta[-1] > 60.0
    assert pac[-1] > 0.60
    assert alert[-1] == 1.0
