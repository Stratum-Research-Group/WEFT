"""
tests/test_microglial_pruning.py
Regression tests for Microglial Complement-Mediated Synaptic Pruning models.
"""

import pytest
import torch
from weft.kernels.fused_microglial_pruning_triton import fused_microglial_pruning_triton


def test_microglial_pruning_triton_cpu_fallback():
    cd47 = torch.tensor([15.0, 50.0, 85.0], dtype=torch.float32)
    c1q = torch.tensor([80.0, 50.0, 30.0], dtype=torch.float32)

    c3b, cr3, survival, ei = fused_microglial_pruning_triton(cd47, c1q)

    # Low CD47 + High C1q: high opsonization, high CR3, low survival
    assert c3b[0].item() > 60.0
    assert cr3[0].item() > 65.0
    assert survival[0].item() < 60.0

    # High CD47 + Low C1q: low opsonization, low CR3, high survival
    assert c3b[2].item() < 25.0
    assert cr3[2].item() < 40.0
    assert survival[2].item() >= 70.0


def test_microglial_pruning_monotonic_protection():
    cd47_sweep = torch.linspace(10.0, 95.0, 30)
    c1q_fixed = torch.full_like(cd47_sweep, 50.0)

    c3b, cr3, survival, ei = fused_microglial_pruning_triton(cd47_sweep, c1q_fixed)

    # C3b opsonization must decrease with increasing CD47
    diff_c3b = torch.diff(c3b)
    assert torch.all(diff_c3b <= 1e-4), "C3b opsonization did not decrease with CD47"

    # Spine survival must increase with increasing CD47
    diff_surv = torch.diff(survival)
    assert torch.all(diff_surv >= -1e-4), "Spine survival did not increase with CD47"
