import pytest
import numpy as np

def test_pulse_to_step_conversion():
    """Verify linear scaling of tonic step firing rate."""
    angles = [0.0, 10.0, 20.0, 30.0, 40.0]
    steps = [60.0 + a * 4.5 for a in angles]
    for i in range(len(steps) - 1):
        assert steps[i] < steps[i+1]
    assert steps[0] == 60.0
    assert steps[-1] == 240.0

def test_leaky_integrator_centripetal_drift():
    """Verify that longer time constants reduce drift velocity."""
    angle = 20.0
    taus = [2.0, 5.0, 10.0, 25.0]
    drifts = [angle / tau for tau in taus]
    for i in range(len(drifts) - 1):
        assert drifts[i] > drifts[i+1]
    assert np.isclose(drifts[-1], 0.80, atol=1e-5)

def test_healthy_drift_preserves_fixation():
    """Verify healthy integrator time constant keeps drift under nystagmus threshold."""
    angle = 25.0
    tau_healthy = 25.0
    drift = angle / tau_healthy
    assert drift <= 1.0 # 1.0 deg/s threshold for stable fixation
