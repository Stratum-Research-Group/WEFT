import pytest
import numpy as np

def test_d1_facilitation_by_dopamine():
    """Verify D1 dSPN activity increases with dopamine concentration."""
    cortex = 80.0
    da_levels = [-0.8, -0.4, 0.0, 0.4, 0.8]
    d1_acts = [cortex * max(0.1, 1.0 + da * 0.8) for da in da_levels]
    for i in range(len(d1_acts) - 1):
        assert d1_acts[i] < d1_acts[i+1]

def test_d2_suppression_by_dopamine():
    """Verify D2 iSPN activity decreases with dopamine concentration."""
    cortex = 80.0
    da_levels = [-0.8, -0.4, 0.0, 0.4, 0.8]
    d2_acts = [cortex * max(0.1, 1.0 - da * 0.7) for da in da_levels]
    for i in range(len(d2_acts) - 1):
        assert d2_acts[i] > d2_acts[i+1]

def test_action_selection_probability_monotonicity():
    """Verify opponent contrast drives monotonic sigmoid action probability."""
    cortex = 75.0
    da_levels = [-0.5, 0.0, 0.5]
    probs = []
    for da in da_levels:
        d1 = cortex * (1.0 + da * 0.8)
        d2 = cortex * (1.0 - da * 0.7)
        prob = 1.0 / (1.0 + np.exp(-(d1 - d2) / 25.0))
        probs.append(prob)

    assert probs[0] < probs[1] < probs[2]
    assert probs[0] < 0.5 < probs[2]
