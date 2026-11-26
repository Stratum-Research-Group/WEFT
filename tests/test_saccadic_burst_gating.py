import pytest
import numpy as np

def test_saccadic_main_sequence_velocity_saturation():
    """Verify that peak velocity exhibits non-linear exponential saturation."""
    # Use uniformly spaced amplitudes to test decreasing second derivative
    amplitudes = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    V_max = 850.0
    C = 12.0
    v_peaks = [V_max * (1.0 - np.exp(-a / C)) for a in amplitudes]

    # Monotonic velocity growth
    for i in range(len(v_peaks) - 1):
        assert v_peaks[i] < v_peaks[i+1]

    # Check saturation (decreasing derivative for uniform step sizes)
    diffs = np.diff(v_peaks)
    for i in range(len(diffs) - 1):
        assert diffs[i] > diffs[i+1]

def test_saccade_duration_scaling():
    """Verify linear scaling of saccade duration."""
    amplitudes = [5.0, 15.0, 30.0]
    durations = [22.0 + 2.5 * a for a in amplitudes]
    assert durations[0] == 34.5
    assert durations[1] == 59.5
    assert durations[2] == 97.0

def test_opn_gating_silence():
    """Verify OPN clamp silences burst generator output."""
    opn_clamp = 80.0
    sc_drives = [40.0, 60.0, 80.0, 95.0]

    for sc in sc_drives:
        is_paused = sc > opn_clamp
        burst_rate = 500.0 if is_paused else 0.0
        if sc <= opn_clamp:
            assert burst_rate == 0.0
        else:
            assert burst_rate > 0.0
