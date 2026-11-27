"""
tests/test_striatal_d1_d2.py
Regression tests for Striatal MSN D1/D2 Dopaminergic Plasticity models.
"""

import pytest
import torch
from weft.kernels.fused_striatal_d1_d2_triton import fused_striatal_d1_d2_triton


def test_striatal_d1_d2_triton_cpu_fallback():
    da = torch.tensor([60.0, 500.0, 850.0, 1800.0], dtype=torch.float32)
    glut = torch.tensor([75.0, 75.0, 75.0, 75.0], dtype=torch.float32)

    pka, cn, d1, d2, ratio = fused_striatal_d1_d2_triton(da, glut)

    # Parkinsonian dopamine depletion ([DA] = 60 nM): low D1 PKA, high D2 Calcineurin, ratio < 1.00
    assert pka[0].item() < 1.2
    assert cn[0].item() > 3.0
    assert ratio[0].item() < 1.00

    # Normal physiological phasic burst ([DA] = 850 nM): direct pathway dominance, ratio >= 1.50
    assert pka[2].item() > 2.8
    assert cn[2].item() < 1.8
    assert d1[2].item() > d2[2].item()
    assert 1.50 <= ratio[2].item() <= 4.50


def test_striatal_d1_d2_monotonicity():
    da_sweep = torch.linspace(50.0, 1500.0, 30)
    glut_fixed = torch.full_like(da_sweep, 75.0)

    pka, cn, d1, d2, ratio = fused_striatal_d1_d2_triton(da_sweep, glut_fixed)

    # D1 PKA and Direct Gain must increase monotonically with dopamine
    diff_pka = torch.diff(pka)
    assert torch.all(diff_pka >= -1e-4), "D1 PKA did not increase monotonically with [DA]"

    # D2 Calcineurin and Indirect Gain must decrease monotonically with dopamine
    diff_cn = torch.diff(cn)
    assert torch.all(diff_cn <= 1e-4), "D2 Calcineurin did not decrease monotonically with [DA]"

    # D1/D2 gain ratio must increase monotonically with dopamine
    diff_ratio = torch.diff(ratio)
    assert torch.all(diff_ratio >= -1e-4), "Gain ratio did not increase monotonically with [DA]"
