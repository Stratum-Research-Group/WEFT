import pytest
import numpy as np
from scripts.sim_dentato_thalamic_sync import simulate_dentato_thalamic_synchronization

def test_dentato_thalamic_synchronization_kinematics():
    """Verify that ascending dentate bursts facilitate VL thalamus and M1 Layer V with high amplitude."""
    res = simulate_dentato_thalamic_synchronization(
        movement_onset_ms=100.0,
        peak_dentate_burst_hz=190.0
    )
    
    # 1. Peak dentate rate must reach intended burst range (> 180 Hz)
    assert res["peak_dentate_rate_hz"] > 180.0, f"Dentate burst insufficient: {res['peak_dentate_rate_hz']} Hz"
    
    # 2. Peak VL thalamic EPSP must exceed 20 mV
    assert res["peak_vl_epsp_mv"] > 20.0, f"VL thalamic EPSP too low: {res['peak_vl_epsp_mv']} mV"
    
    # 3. Peak M1 Layer V facilitation must exceed 15 mV
    assert res["peak_m1_facil_mv"] > 15.0, f"M1 Layer V facilitation too low: {res['peak_m1_facil_mv']} mV"
    
    # 4. Phase-locking coherence during burst must be positive
    assert res["mean_burst_coherence"] > 0.25, f"Coherence too low: {res['mean_burst_coherence']}"

def test_dentate_rate_monotonicity():
    """Verify that higher dentate discharge rates produce higher VL thalamic EPSPs."""
    res_low = simulate_dentato_thalamic_synchronization(peak_dentate_burst_hz=100.0)
    res_high = simulate_dentato_thalamic_synchronization(peak_dentate_burst_hz=220.0)
    
    assert res_high["peak_vl_epsp_mv"] > res_low["peak_vl_epsp_mv"]
    assert res_high["peak_m1_facil_mv"] > res_low["peak_m1_facil_mv"]
