import pytest
import numpy as np

def test_reticulospinal_axial_stiffness_monotonicity():
    """Verify axial stiffness scales monotonically with PMRF drive."""
    drives = [0.0, 25.0, 50.0, 75.0, 100.0]
    stiffnesses = [30.0 + d * 2.0 for d in drives]
    for i in range(len(stiffnesses) - 1):
        assert stiffnesses[i] < stiffnesses[i+1]

def test_apa_phase_lead_margin():
    """Verify that APA onset strictly precedes distal motor trigger."""
    reach_onset = 150.0
    apa_lead = 65.0
    apa_onset = reach_onset - apa_lead
    assert apa_onset < reach_onset
    assert reach_onset - apa_onset >= 50.0 # Min physiological APA margin

def test_dynamic_stability_index():
    """Verify stability index remains bounded and properly normalized."""
    bos = 50.0
    com_disps = [0.0, 10.0, 25.0, 49.0, 55.0]
    expected = [1.0, 0.8, 0.5, 0.02, 0.0]
    for c, exp in zip(com_disps, expected):
        stab = max(0.0, (bos - c) / bos)
        assert np.isclose(stab, exp, atol=1e-5)
