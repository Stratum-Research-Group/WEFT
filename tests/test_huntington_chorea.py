"""
Regression Test Suite: Milestone Target #400
Huntington's Disease: Striatal D2-SPN Degeneration & Hyperkinetic Chorea
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group
"""

import pytest
import numpy as np
import torch

from weft.kernels.fused_huntington_chorea_triton import compute_huntington_chorea_triton


def test_huntington_healthy_baseline():
    """Verify that 100% D2-SPN survival maintains restrained GPe firing, active STN brake, and zero chorea."""
    surv = torch.tensor([1.0], dtype=torch.float32)
    resc = torch.tensor([0.0], dtype=torch.float32)

    gpe, stn, thal, chorea = compute_huntington_chorea_triton(surv, resc)

    assert float(gpe[0]) == pytest.approx(52.0, rel=1e-3)
    assert float(stn[0]) > 20.0
    assert float(thal[0]) < 30.0
    assert float(chorea[0]) == 0.0


def test_d2_spn_degeneration_causes_chorea():
    """Verify that severe D2-SPN loss (15% survival) triggers GPe hyperactivity, STN silencing, and severe chorea."""
    surv = torch.tensor([0.15], dtype=torch.float32)
    resc = torch.tensor([0.0], dtype=torch.float32)

    gpe, stn, thal, chorea = compute_huntington_chorea_triton(surv, resc)

    assert float(gpe[0]) > 85.0
    assert float(stn[0]) < 10.0  # STN brake collapse
    assert float(thal[0]) > 80.0  # Thalamic disinhibition
    assert float(chorea[0]) > 7.0  # Severe chorea (> 7/10)


def test_degeneration_monotonicity():
    """Verify that progressive loss of D2-SPNs monotonically increases GPe firing and chorea."""
    surv_levels = torch.linspace(1.0, 0.1, 20)
    resc = torch.zeros(20, dtype=torch.float32)

    gpe, stn, thal, chorea = compute_huntington_chorea_triton(surv_levels, resc)
    chorea_np = chorea.cpu().numpy()

    # Progressive loss (decreasing survival) must lead to non-decreasing chorea
    diffs = np.diff(chorea_np)
    assert np.all(diffs >= -1e-5), "Chorea score must increase monotonically as D2-SPN survival decreases"


def test_gpe_gaba_rescue_reduces_chorea():
    """Verify that GPe GABAergic rescue restores the STN brake and suppresses chorea below 2.5/10."""
    surv_depleted = torch.tensor([0.15, 0.15], dtype=torch.float32)
    resc_conditions = torch.tensor([0.0, 0.85], dtype=torch.float32)

    gpe, stn, thal, chorea = compute_huntington_chorea_triton(surv_depleted, resc_conditions)

    # Unrescued
    assert float(chorea[0]) > 7.0
    # Rescued
    assert float(chorea[1]) < 2.5
    assert float(stn[1]) > float(stn[0])  # STN brake restored
    assert float(gpe[1]) < float(gpe[0])  # GPe hyperactivity suppressed
