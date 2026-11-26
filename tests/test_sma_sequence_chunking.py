import pytest
import numpy as np
from scripts.sim_sma_sequence_chunking import simulate_sma_sequence_execution

def test_sma_sequence_chunking_execution():
    """Verify that a 4-step sequence executes with high-fidelity chunk transitions and ramp timing."""
    res = simulate_sma_sequence_execution(step_durations_ms=[200.0, 150.0, 180.0, 220.0])
    
    # 1. Total duration must match sum of step durations
    assert abs(res["total_duration_ms"] - 750.0) < 1e-3
    assert res["n_chunks"] == 4
    
    # 2. Verified transition occurrences
    assert len(res["transition_times_ms"]) == 4
    assert abs(res["transition_times_ms"][0] - 200.0) < 2.0
    assert abs(res["transition_times_ms"][1] - 350.0) < 2.0
    assert abs(res["transition_times_ms"][2] - 530.0) < 2.0
    
    # 3. Peak SMA ramp timing rate in physiological range (150 - 180 Hz)
    assert 150.0 <= res["peak_sma_rate_hz"] <= 180.0, f"Peak SMA rate out of bounds: {res['peak_sma_rate_hz']} Hz"
    
    # 4. Pre-SMA switching burst at chunk boundary (> 150 Hz)
    assert res["peak_pre_sma_rate_hz"] > 150.0, f"Pre-SMA switching burst insufficient: {res['peak_pre_sma_rate_hz']} Hz"
    
    # 5. M1 trigger duty cycle should allow motor execution during most of the step (> 75%)
    assert res["mean_m1_permissiveness"] > 0.75, f"M1 permissiveness too low: {res['mean_m1_permissiveness']}"

def test_sma_timing_monotonicity():
    """Verify that within each chunk, SMA proper firing rate increases monotonically."""
    res = simulate_sma_sequence_execution(step_durations_ms=[100.0])
    rates = res["sma_rate_hist"]
    # Check monotonicity
    diffs = np.diff(rates)
    assert np.all(diffs >= -1e-6), "SMA timing rate is not monotonic"
