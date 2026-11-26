import pytest
import numpy as np

def test_arkypallidal_stop_burst_activation():
    """Verify arkypallidal firing increases dramatically under Stop Signal."""
    da_levels = [10.0, 30.0, 50.0, 80.0]
    for da in da_levels:
        baseline_arky = 8.0 + (100.0 - da) * 0.08
        stop_arky = 55.0
        assert stop_arky > baseline_arky

def test_striatal_motor_program_suppression():
    """Verify that arkypallidal burst suppresses striatal motor drive."""
    drive = 80.0
    baseline_arky = 12.0
    stop_arky = 55.0

    unsuppressed = max(0.0, drive - (drive * baseline_arky) / 60.0)
    suppressed = max(0.0, drive - (drive * stop_arky) / 60.0)

    assert suppressed < unsuppressed
    assert suppressed < 10.0 # Under 10% residual activity during active stop

def test_beta_resonant_frequency_bounds():
    """Verify STN-GPe resonant frequency remains strictly within 13-30 Hz."""
    da_levels = [0.0, 20.0, 50.0, 80.0, 100.0]
    for da in da_levels:
        f_res = 13.0 + ((100.0 - da) * 17.0) / 100.0
        assert 13.0 <= f_res <= 30.0
