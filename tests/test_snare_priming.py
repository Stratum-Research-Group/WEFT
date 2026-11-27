"""
tests/test_snare_priming.py
Regression tests for Presynaptic SNARE Complex Assembly & Munc18-1 Priming models.
"""

import pytest
import torch
from weft.kernels.fused_snare_priming_triton import fused_snare_priming_triton


def test_snare_priming_triton_cpu_fallback():
    munc13 = torch.tensor([15.0, 50.0, 90.0], dtype=torch.float32)
    ca = torch.tensor([5.0, 10.0, 18.0], dtype=torch.float32)

    open_syn, zip_val, rrp, jitter = fused_snare_priming_triton(munc13, ca)

    # Low Munc13: low open syntaxin, low RRP, high jitter
    assert open_syn[0].item() < 25.0
    assert rrp[0].item() < 60.0
    assert jitter[0].item() > 350.0

    # High Munc13 + adequate Ca2+: high open syntaxin, high RRP, sub-250 us jitter
    assert open_syn[2].item() > 75.0
    assert rrp[2].item() >= 80.0
    assert jitter[2].item() <= 250.0


def test_snare_priming_monotonicity():
    munc13_sweep = torch.linspace(10.0, 95.0, 30)
    ca_fixed = torch.full_like(munc13_sweep, 15.0)

    open_syn, zip_val, rrp, jitter = fused_snare_priming_triton(munc13_sweep, ca_fixed)

    # RRP occupancy must be monotonically non-decreasing
    diff_rrp = torch.diff(rrp)
    assert torch.all(diff_rrp >= -1e-4), "RRP occupancy did not increase monotonically with Munc13"

    # Jitter must be monotonically non-increasing
    diff_jit = torch.diff(jitter)
    assert torch.all(diff_jit <= 1e-4), "Release jitter did not decrease monotonically with Munc13"
