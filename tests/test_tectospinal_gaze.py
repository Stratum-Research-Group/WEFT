import pytest
import numpy as np
from scripts.sim_tectospinal_gaze_orienting import simulate_tectospinal_gaze_shift

def test_tectospinal_gaze_orienting_kinematics():
    """Verify that a 30 degree gaze shift satisfies target acquisition and eye-head decomposition."""
    res = simulate_tectospinal_gaze_shift(target_eccentricity_deg=30.0, head_threshold_deg=15.0)
    
    # 1. Target acquisition accuracy (residual error < 1.0 deg)
    assert res["final_error_deg"] < 1.0, f"Final error too large: {res['final_error_deg']}"
    
    # 2. DLSC peak burst firing rate (> 500 Hz for 30 deg saccade)
    assert res["peak_burst_hz"] > 500.0, f"Peak burst insufficient: {res['peak_burst_hz']}"
    
    # 3. Eye velocity reaches saccadic main sequence speed (> 400 deg/s)
    assert res["peak_eye_vel_deg_s"] > 400.0, f"Eye velocity too low: {res['peak_eye_vel_deg_s']}"
    
    # 4. Head movement recruited above 15 deg threshold
    assert res["head_contribution_deg"] > 4.0, f"Head contribution too low: {res['head_contribution_deg']}"
    assert res["peak_head_vel_deg_s"] > 50.0, f"Head velocity too low: {res['peak_head_vel_deg_s']}"
    
    # 5. Latency to target acquisition < 150 ms
    assert res["acq_latency_ms"] < 150.0, f"Acquisition took too long: {res['acq_latency_ms']}"

def test_head_threshold_gating():
    """Verify that for targets below the 15 deg threshold, head contribution is negligible."""
    res_small = simulate_tectospinal_gaze_shift(target_eccentricity_deg=10.0, head_threshold_deg=15.0)
    
    # Target acquisition accurate
    assert res_small["final_error_deg"] < 1.0
    
    # Head contribution should remain minimal / zero
    assert res_small["head_contribution_deg"] < 0.5, f"Head recruited prematurely: {res_small['head_contribution_deg']}"
    assert res_small["peak_head_vel_deg_s"] < 5.0, f"Head velocity unexpected: {res_small['peak_head_vel_deg_s']}"

def test_eccentricity_monotonicity():
    """Verify that larger eccentricities produce higher burst rates and higher peak velocities."""
    res_15 = simulate_tectospinal_gaze_shift(target_eccentricity_deg=15.0)
    res_35 = simulate_tectospinal_gaze_shift(target_eccentricity_deg=35.0)
    
    assert res_35["peak_burst_hz"] > res_15["peak_burst_hz"]
    assert res_35["peak_eye_vel_deg_s"] >= res_15["peak_eye_vel_deg_s"]
    assert res_35["peak_head_vel_deg_s"] > res_15["peak_head_vel_deg_s"]
