"""
tests/test_hippocampal_swr.py
Regression tests for Hippocampal CA1 Sharp Wave-Ripple (SWR) models.
"""

import pytest
import torch
from weft.kernels.fused_hippocampal_swr_triton import fused_hippocampal_swr_triton


def test_hippocampal_swr_triton_cpu_fallback():
    pv = torch.tensor([30.0, 60.0, 85.0], dtype=torch.float32)
    ca3 = torch.tensor([75.0, 75.0, 80.0], dtype=torch.float32)

    freq, plv, fidelity, spikes = fused_hippocampal_swr_triton(pv, ca3)

    # Low PV drive: low frequency, low PLV, low fidelity
    assert freq[0].item() < 170.0
    assert plv[0].item() < 0.60
    assert fidelity[0].item() < 60.0

    # High PV drive + CA3 input: 150-250 Hz ripple, PLV >= 0.80, fidelity >= 80%
    assert 180.0 <= freq[2].item() <= 220.0
    assert plv[2].item() >= 0.80
    assert fidelity[2].item() >= 80.0
    assert spikes[2].item() >= 5.0


def test_hippocampal_swr_monotonicity():
    pv_sweep = torch.linspace(20.0, 95.0, 30)
    ca3_fixed = torch.full_like(pv_sweep, 75.0)

    freq, plv, fidelity, spikes = fused_hippocampal_swr_triton(pv_sweep, ca3_fixed)

    # Frequency must increase monotonically with PV drive
    diff_freq = torch.diff(freq)
    assert torch.all(diff_freq >= -1e-4), "Ripple frequency did not increase with PV drive"

    # PLV must increase monotonically with PV drive
    diff_plv = torch.diff(plv)
    assert torch.all(diff_plv >= -1e-4), "Phase-locking value did not increase with PV drive"
