import pytest
import numpy as np

def test_vor_adaptive_gain_convergence():
    """Verify that VOR gain converges monotonically toward magnifying target."""
    target_gain = 1.50
    current_gain = 1.00
    learning_rate = 0.05
    history = [current_gain]

    for _ in range(50):
        current_gain += learning_rate * (target_gain - current_gain)
        history.append(current_gain)

    # Monotonic increase
    for i in range(len(history) - 1):
        assert history[i] < history[i+1]
    assert np.isclose(history[-1], target_gain, atol=0.05)

def test_purkinje_ltd_weight_decrement():
    """Verify that Purkinje synaptic weight decreases monotonically during gain elevation."""
    w = 1.0
    weights = [w]
    for _ in range(10):
        dw = 0.05
        w = max(0.0, w - dw)
        weights.append(w)

    for i in range(len(weights) - 1):
        assert weights[i] >= weights[i+1]
    assert weights[-1] < weights[0]

def test_climbing_fiber_error_attenuation():
    """Verify climbing fiber firing rate returns to baseline as error diminishes."""
    slips = [50.0, 25.0, 10.0, 1.0]
    cf_rates = [1.0 + (s / 50.0) * 3.5 for s in slips]
    for i in range(len(cf_rates) - 1):
        assert cf_rates[i] > cf_rates[i+1]
    assert np.isclose(cf_rates[-1], 1.07, atol=0.1)
