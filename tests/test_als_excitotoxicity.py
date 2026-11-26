"""
Regression Test Suite: Target #401
ALS: Corticomotoneuronal Excitotoxicity & Renshaw Recurrent Disinhibition
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group
"""

import pytest
import numpy as np
import torch

from weft.kernels.fused_als_excitotoxicity_triton import compute_als_excitotoxicity_triton


def test_als_healthy_homeostasis():
    """Verify that 100% EAAT2 and Renshaw integrity clamp glutamate and maintain normal motoneuron rate."""
    eaat2 = torch.tensor([1.0], dtype=torch.float32)
    renshaw = torch.tensor([1.0], dtype=torch.float32)
    riluzole = torch.tensor([0.0], dtype=torch.float32)
    glycine = torch.tensor([0.0], dtype=torch.float32)

    glu, rate, ren_gain, fasc = compute_als_excitotoxicity_triton(eaat2, renshaw, riluzole, glycine)

    assert float(glu[0]) == pytest.approx(1.2, rel=1e-3)
    assert float(ren_gain[0]) == pytest.approx(0.80, rel=1e-3)
    assert float(rate[0]) <= 15.0
    assert float(fasc[0]) == pytest.approx(0.1, rel=1e-3)


def test_eaat2_loss_and_renshaw_degeneration_cause_excitotoxicity():
    """Verify that EAAT2 loss and Renshaw degeneration produce runaway firing and high fasciculation rates."""
    eaat2 = torch.tensor([0.15], dtype=torch.float32)
    renshaw = torch.tensor([0.20], dtype=torch.float32)
    riluzole = torch.tensor([0.0], dtype=torch.float32)
    glycine = torch.tensor([0.0], dtype=torch.float32)

    glu, rate, ren_gain, fasc = compute_als_excitotoxicity_triton(eaat2, renshaw, riluzole, glycine)

    assert float(glu[0]) > 10.0  # Glutamate accumulation > 10 uM
    assert float(rate[0]) > 45.0  # Runaway firing > 45 Hz
    assert float(fasc[0]) > 3.0  # Pathological fasciculations > 3 Hz


def test_excitotoxicity_monotonicity():
    """Verify that progressive loss of EAAT2 transporter monotonically drives glutamate accumulation."""
    eaat2_levels = torch.linspace(1.0, 0.1, 20)
    renshaw = torch.ones(20, dtype=torch.float32) * 0.5
    riluzole = torch.zeros(20, dtype=torch.float32)
    glycine = torch.zeros(20, dtype=torch.float32)

    glu, rate, _, _ = compute_als_excitotoxicity_triton(eaat2_levels, renshaw, riluzole, glycine)
    glu_np = glu.cpu().numpy()

    # Decreasing EAAT2 must lead to non-decreasing synaptic glutamate
    diffs = np.diff(glu_np)
    assert np.all(diffs >= -1e-5), "Glutamate concentration must increase monotonically as EAAT2 decreases"


def test_dual_riluzole_glycine_rescue():
    """Verify that dual therapy (Riluzole + Glycine agonist) stabilizes motoneuron rate <= 25 Hz."""
    eaat2 = torch.tensor([0.15, 0.15], dtype=torch.float32)
    renshaw = torch.tensor([0.20, 0.20], dtype=torch.float32)
    riluzole = torch.tensor([0.0, 0.85], dtype=torch.float32)
    glycine = torch.tensor([0.0, 0.80], dtype=torch.float32)

    glu, rate, ren_gain, fasc = compute_als_excitotoxicity_triton(eaat2, renshaw, riluzole, glycine)

    # Unrescued
    assert float(rate[0]) > 45.0
    assert float(fasc[0]) > 3.0

    # Rescued
    assert float(rate[1]) <= 25.0
    assert float(fasc[1]) < 0.5
    assert float(glu[1]) < float(glu[0])
    assert float(ren_gain[1]) > float(ren_gain[0])
