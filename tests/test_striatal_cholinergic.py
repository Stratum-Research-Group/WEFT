"""
Regression Test Suite: Target #398
Striatal Cholinergic Interneuron (ChI) Hyper-Rebound & Dystonic Co-Contraction
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group
"""

import pytest
import numpy as np
import torch

from weft.kernels.fused_striatal_cholinergic_triton import compute_striatal_cholinergic_triton


def test_chi_baseline_tonic_firing():
    """Verify that full Dopamine D2 tone preserves healthy baseline firing rate and reciprocal inhibition."""
    d2 = torch.tensor([1.0], dtype=torch.float32)
    m4 = torch.tensor([0.0], dtype=torch.float32)

    rate, ach, ag, ant, cci = compute_striatal_cholinergic_triton(d2, m4)

    assert float(rate[0]) == pytest.approx(4.5, rel=1e-3)
    assert float(ach[0]) == pytest.approx(18.9, rel=1e-3)
    assert float(ant[0]) == pytest.approx(8.0, rel=1e-3)
    assert float(cci[0]) < 0.45


def test_d2_depletion_triggers_dystonic_hyper_rebound():
    """Verify that severe D2 depletion triggers hyper-rebound firing rate and pathological co-contraction."""
    d2 = torch.tensor([0.05], dtype=torch.float32)
    m4 = torch.tensor([0.0], dtype=torch.float32)

    rate, ach, ag, ant, cci = compute_striatal_cholinergic_triton(d2, m4)

    # Rebound rate must exceed 25 Hz and co-contraction must exceed 50%
    assert float(rate[0]) > 25.0
    assert float(ach[0]) > 100.0
    assert float(ant[0]) > 12.0
    assert float(cci[0]) > 0.50


def test_rebound_attenuation_monotonicity():
    """Verify monotonic suppression of hyper-rebound across increasing D2 tone."""
    d2_levels = torch.linspace(0.0, 1.0, 20)
    m4 = torch.zeros(20, dtype=torch.float32)

    rates, _, _, _, _ = compute_striatal_cholinergic_triton(d2_levels, m4)
    rates_np = rates.cpu().numpy()

    # Verify strict non-increasing monotonic behavior
    diffs = np.diff(rates_np)
    assert np.all(diffs <= 1e-5), "ChI rebound firing rate must decrease monotonically with increasing D2 tone"


def test_m4_pam_anticholinergic_rescue():
    """Verify that M4 positive allosteric modulation rescues co-contraction in dopamine-depleted state."""
    d2_depleted = torch.tensor([0.05, 0.05], dtype=torch.float32)
    m4_conditions = torch.tensor([0.0, 0.85], dtype=torch.float32)  # Unrescued vs M4 PAM

    rates, achs, ags, ants, ccis = compute_striatal_cholinergic_triton(d2_depleted, m4_conditions)

    # Unrescued should exhibit dystonic co-contraction
    assert float(ccis[0]) > 0.50
    # M4 rescued should restore CCI back into safe physiological window (< 0.45)
    assert float(ccis[1]) < 0.45
    # Antagonist leakage must be significantly reduced
    assert float(ants[1]) < float(ants[0])
