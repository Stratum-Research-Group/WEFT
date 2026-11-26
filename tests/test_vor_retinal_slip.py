import pytest
import numpy as np

def test_vor_counter_rotational_eye_velocity():
    """Verify that compensatory eye velocity opposes head rotation."""
    head_velocities = [-200.0, -100.0, 0.0, 100.0, 200.0]
    gain = 0.98
    for h in head_velocities:
        eye_vel = -h * gain
        if h > 0:
            assert eye_vel < 0
        elif h < 0:
            assert eye_vel > 0
        else:
            assert eye_vel == 0

def test_retinal_slip_gain_dependence():
    """Verify that unity gain minimizes retinal slip."""
    head_vel = 150.0
    gains = [0.80, 0.90, 0.98, 1.00]
    slips = [abs(head_vel + (-head_vel * g)) for g in gains]
    # Slips should decrease as gain approaches 1.0
    for i in range(len(slips) - 1):
        assert slips[i] > slips[i+1]
    assert np.isclose(slips[-1], 0.0, atol=1e-5)

def test_acuity_preservation_threshold():
    """Verify that normal physiological VOR keeps slip within acuity boundary."""
    head_vel = 100.0
    gain = 0.985
    slip = abs(head_vel + (-head_vel * gain))
    assert slip < 2.0 # Standard threshold in deg/s for undisturbed acuity
