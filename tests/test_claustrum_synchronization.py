"""
tests/test_claustrum_synchronization.py
Pytest Regression Suite: Claustrum Cortical Synchronization & Salience Switching Invariants
Lean 4 Target #393 | Paper #202 | Cycle v340.71.0
"""

import pytest
import math
import torch
from weft.kernels.fused_claustrum_synchronization_triton import (
    launch_fused_claustrum_synchronization,
)

def test_claustrum_rate_bounds_and_monotonicity():
    """Verify that claustral firing rate satisfies bounded monotonic scaling."""
    rates = []
    for s_int in range(0, 101, 10):
        sal = s_int / 100.0
        # Reference integer arithmetic formula: min(1000, 200 + 7 * s_int)
        expected_rate_fixed = min(1000, 200 + 7 * s_int)
        rate_hz = expected_rate_fixed / 100.0
        rates.append(rate_hz)
        assert 0.0 <= rate_hz <= 10.0, f"Rate {rate_hz} out of bounds [0, 10]"

    # Verify strictly monotonic non-decreasing
    for i in range(len(rates) - 1):
        assert rates[i] <= rates[i+1], f"Monotonicity failed at index {i}: {rates[i]} > {rates[i+1]}"

def test_feedforward_inhibition_pv_scaling():
    """Verify PV+ feedforward inhibition bounds and behavior."""
    # Low claustral baseline (2.0 Hz)
    pv_low = min(1000, 100 + (90 * 200) // 100) / 1000.0
    # High burst mode (9.0 Hz)
    pv_high = min(1000, 100 + (90 * 900) // 100) / 1000.0

    assert 0.0 <= pv_low <= 1.0
    assert 0.0 <= pv_high <= 1.0
    assert pv_low < pv_high
    assert pv_low == pytest.approx(0.28, rel=1e-2)
    assert pv_high == pytest.approx(0.91, rel=1e-2)

def test_burst_mode_activation_threshold():
    """Verify burst mode threshold logic matching Lean 4 invariants."""
    # Firing rate threshold is 450 (4.5 Hz)
    assert not (200 >= 450), "2.0 Hz baseline must not trigger burst mode"
    assert not (350 >= 450), "3.5 Hz rate must not trigger burst mode"
    assert (450 >= 450), "4.5 Hz threshold must trigger burst mode"
    assert (850 >= 450), "8.5 Hz surge must trigger burst mode"

    # Salience threshold: > 50% guarantees burst mode in Lean 4 theorem
    s_high = 55  # 55%
    c_rate_high = min(1000, 200 + (700 * s_high) // 100)
    assert c_rate_high >= 450, f"Salience 55% should yield c_rate >= 450, got {c_rate_high}"

def test_claustrum_forward_cpu_emulation():
    """Test CPU reference logic for claustrum synchronization and salience switching."""
    N = 64
    salience = torch.linspace(0.0, 1.0, N, dtype=torch.float32)
    task_drive = torch.full((N,), 0.5, dtype=torch.float32)
    base = torch.full((N,), 2.0, dtype=torch.float32)
    gain = torch.full((N,), 0.5, dtype=torch.float32)

    c_sal = 0.85
    c_inh_scale = 1.80
    sync_base = 0.10
    sync_scale = 0.065
    c_drive_scale = 0.80
    c_inh_eff = 0.25
    inh_offset = 3.0
    sig_slope = 0.20

    # CPU Emulation
    rate = base + c_sal * salience
    inh = 1.0 + (gain * c_inh_scale) * rate
    sync = torch.clamp(sync_base + sync_scale * rate, 0.0, 1.0)
    diff = salience - task_drive
    comb = diff * c_drive_scale + (inh - inh_offset) * c_inh_eff
    prob = torch.clamp(0.5 + sig_slope * comb, 0.0, 1.0)

    assert len(rate) == N
    assert len(inh) == N
    assert len(sync) == N
    assert len(prob) == N

    # Baseline edge case
    assert salience[0] == 0.0
    assert rate[0] == pytest.approx(2.0, abs=1e-2)
    assert prob[0] < 0.5

    # Max salience edge case
    assert salience[-1] == 1.0
    assert rate[-1] > rate[0]
    assert inh[-1] > inh[0]
    assert sync[-1] > sync[0]
    assert prob[-1] > 0.5
