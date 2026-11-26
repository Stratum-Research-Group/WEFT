import pytest
import numpy as np
from scripts.sim_stn_hyperdirect_stop import simulate_hyperdirect_stop_signal

def test_hyperdirect_stop_signal_cancellation():
    """Verify that a stop signal presented with SSD = 100 ms successfully cancels motor output with SSRT < 25 ms."""
    res = simulate_hyperdirect_stop_signal(
        stop_signal_delay_ms=100.0,
        inject_stop_signal=True
    )
    
    # 1. Action must be successfully aborted before completion
    assert res["action_aborted"] is True, "Action was not aborted"
    assert res["action_completed"] is False, "Action executed despite stop signal"
    
    # 2. SSRT must be ultra-fast (< 25 ms)
    assert res["ssrt_ms"] is not None
    assert res["ssrt_ms"] < 25.0, f"SSRT too slow: {res['ssrt_ms']} ms"
    
    # 3. Peak STN and GPi discharge must exceed abort thresholds
    assert res["peak_stn_rate_hz"] > 150.0, f"STN burst too low: {res['peak_stn_rate_hz']} Hz"
    assert res["peak_gpi_rate_hz"] > 140.0, f"GPi pallidal burst too low: {res['peak_gpi_rate_hz']} Hz"

def test_no_stop_signal_execution():
    """Verify that when no stop signal is present, the Go process runs to completion."""
    res_go = simulate_hyperdirect_stop_signal(
        inject_stop_signal=False
    )
    
    assert res_go["action_aborted"] is False
    assert res_go["action_completed"] is True
    assert res_go["completion_time_ms"] is not None
    # STN should remain at baseline
    assert res_go["peak_stn_rate_hz"] == 20.0

def test_stop_signal_monotonicity():
    """Verify that earlier stop signals (lower SSD) abort earlier in time."""
    res_early = simulate_hyperdirect_stop_signal(stop_signal_delay_ms=60.0)
    res_late = simulate_hyperdirect_stop_signal(stop_signal_delay_ms=110.0)
    
    assert res_early["abort_time_ms"] < res_late["abort_time_ms"]
