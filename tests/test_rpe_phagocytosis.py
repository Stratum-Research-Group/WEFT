"""
tests/test_rpe_phagocytosis.py
Regression tests for RPE Photoreceptor Outer Segment (POS) Phagocytosis models.
"""

import pytest
import torch
from weft.kernels.fused_rpe_phagocytosis_triton import fused_rpe_phagocytosis_triton


def test_rpe_phagocytosis_triton_cpu_fallback():
    mertk = torch.tensor([15.0, 50.0, 85.0], dtype=torch.float32)
    gas6 = torch.tensor([80.0, 80.0, 80.0], dtype=torch.float32)

    binding, clearance, a2e, survival = fused_rpe_phagocytosis_triton(mertk, gas6)

    # Low MerTK autophosphorylation (RCS rat defect): clearance collapse, toxic A2E accumulation
    assert clearance[0].item() < 30.0
    assert a2e[0].item() > 70.0
    assert survival[0].item() < 60.0

    # Healthy physiological phagocytosis (MerTK >= 65%): complete clearance, low A2E, high survival
    assert clearance[2].item() >= 95.0
    assert a2e[2].item() <= 20.0
    assert survival[2].item() >= 85.0


def test_rpe_phagocytosis_monotonicity():
    mertk_sweep = torch.linspace(10.0, 95.0, 30)
    gas6_fixed = torch.full_like(mertk_sweep, 80.0)

    binding, clearance, a2e, survival = fused_rpe_phagocytosis_triton(mertk_sweep, gas6_fixed)

    # POS clearance must increase monotonically with MerTK phosphorylation
    diff_clear = torch.diff(clearance)
    assert torch.all(diff_clear >= -1e-4), "POS clearance did not increase monotonically with MerTK"

    # A2E burden must decrease monotonically with MerTK phosphorylation
    diff_a2e = torch.diff(a2e)
    assert torch.all(diff_a2e <= 1e-4), "A2E burden did not decrease monotonically with MerTK"

    # Photoreceptor survival must increase monotonically with MerTK phosphorylation
    diff_surv = torch.diff(survival)
    assert torch.all(diff_surv >= -1e-4), "Photoreceptor survival did not increase with MerTK"
