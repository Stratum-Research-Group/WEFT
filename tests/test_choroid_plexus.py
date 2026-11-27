"""
tests/test_choroid_plexus.py
Regression tests for Choroid Plexus CSF Secretion & Intracranial Pressure (ICP) models.
"""

import pytest
import torch
from weft.kernels.fused_choroid_plexus_triton import fused_choroid_plexus_triton


def test_choroid_plexus_triton_cpu_fallback():
    ca2 = torch.tensor([25.0, 50.0, 85.0], dtype=torch.float32)
    nkcc1 = torch.tensor([25.0, 50.0, 80.0], dtype=torch.float32)

    grad, csf, icp, comp = fused_choroid_plexus_triton(ca2, nkcc1)

    # Inhibited CA-II & NKCC1: suppressed CSF secretion, preserved compliance
    assert csf[0].item() < 0.25
    assert icp[0].item() < 10.5
    assert comp[0].item() > 75.0

    # Physiological baseline: normal osmotic gradient, normal CSF rate (0.35 - 0.40 mL/min)
    assert 18.0 <= grad[2].item() <= 25.0
    assert 0.32 <= csf[2].item() <= 0.45
    assert icp[2].item() <= 15.0
    assert comp[2].item() >= 60.0


def test_choroid_plexus_monotonicity():
    ca2_sweep = torch.linspace(10.0, 95.0, 30)
    nkcc1_fixed = torch.full_like(ca2_sweep, 75.0)

    grad, csf, icp, comp = fused_choroid_plexus_triton(ca2_sweep, nkcc1_fixed)

    # Osmotic gradient must increase monotonically with CA-II activity
    diff_grad = torch.diff(grad)
    assert torch.all(diff_grad >= -1e-4), "Osmotic gradient did not increase with CA-II"

    # CSF production must increase monotonically with CA-II activity
    diff_csf = torch.diff(csf)
    assert torch.all(diff_csf >= -1e-4), "CSF secretion did not increase with CA-II"

    # Compliance reserve must decrease monotonically as ICP increases
    diff_comp = torch.diff(comp)
    assert torch.all(diff_comp <= 1e-4), "Compliance reserve did not decrease as ICP increased"
