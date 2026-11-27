"""
tests/test_purkinje_complex_spike.py
Regression tests for Cerebellar Purkinje Complex Spike & Climbing Fiber LTD models.
"""

import pytest
import torch
from weft.kernels.fused_purkinje_complex_spike_triton import fused_purkinje_complex_spike_triton


def test_purkinje_complex_spike_triton_cpu_fallback():
    cf = torch.tensor([0.2, 1.2, 2.5, 5.0], dtype=torch.float32)
    pf = torch.tensor([80.0, 80.0, 80.0, 80.0], dtype=torch.float32)

    ca, mglur, ltd, error = fused_purkinje_complex_spike_triton(cf, pf)

    # Silenced climbing fiber (CF = 0.2 Hz): low Ca2+, LTD failure, elevated error
    assert ca[0].item() < 1.00
    assert ltd[0].item() < 25.0
    assert error[0].item() > 0.60

    # Physiological learning burst (CF = 2.5 Hz): suprathreshold Ca2+, LTD >= 35%, calibrated error
    assert ca[2].item() >= 1.80
    assert mglur[2].item() > 3.0
    assert ltd[2].item() >= 35.0
    assert error[2].item() <= 0.45


def test_purkinje_complex_spike_monotonicity():
    cf_sweep = torch.linspace(0.5, 4.0, 30)
    pf_fixed = torch.full_like(cf_sweep, 80.0)

    ca, mglur, ltd, error = fused_purkinje_complex_spike_triton(cf_sweep, pf_fixed)

    # Dendritic calcium must increase monotonically with climbing fiber frequency
    diff_ca = torch.diff(ca)
    assert torch.all(diff_ca >= -1e-4), "Dendritic Ca2+ did not increase monotonically with CF frequency"

    # Parallel fiber LTD must increase monotonically with climbing fiber frequency
    diff_ltd = torch.diff(ltd)
    assert torch.all(diff_ltd >= -1e-4), "Parallel fiber LTD did not increase with CF frequency"

    # Motor prediction error must decrease monotonically as LTD increases
    diff_err = torch.diff(error)
    assert torch.all(diff_err <= 1e-4), "Prediction error did not decrease monotonically with LTD"
