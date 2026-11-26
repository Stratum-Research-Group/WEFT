import pytest
import numpy as np
from scripts.sim_corticospinal_digit_grasp import simulate_corticospinal_digit_grasp

def test_corticospinal_fractionated_grasp():
    """Verify that a 12 N index finger force command yields fractionated force with high fidelity."""
    res = simulate_corticospinal_digit_grasp(target_force_n=12.0)
    
    # 1. Force tracking error < 5%
    assert res["tracking_error_pct"] < 5.0, f"Tracking error too high: {res['tracking_error_pct']}%"
    
    # 2. M1 Betz cell firing rate in physiological range (80 - 150 Hz)
    assert 80.0 <= res["peak_m1_rate_hz"] <= 150.0, f"M1 rate out of range: {res['peak_m1_rate_hz']} Hz"
    
    # 3. Non-target spillover force < 1.0 N
    assert res["final_spillover_force_n"] < 1.0, f"Excessive spillover force: {res['final_spillover_force_n']} N"
    
    # 4. Fractionation selectivity index > 85%
    assert res["final_fractionation_index"] > 0.85, f"Low fractionation index: {res['final_fractionation_index']}"

def test_corticospinal_force_monotonicity():
    """Verify that larger target force commands scale M1 firing rate and contractile force monotonically."""
    res_5 = simulate_corticospinal_digit_grasp(target_force_n=5.0)
    res_15 = simulate_corticospinal_digit_grasp(target_force_n=15.0)
    
    assert res_15["peak_m1_rate_hz"] > res_5["peak_m1_rate_hz"]
    assert res_15["final_target_force_n"] > res_5["final_target_force_n"]
    assert res_15["final_spillover_force_n"] > res_5["final_spillover_force_n"]

def test_fractionation_preservation_under_moderate_force():
    """Verify that even under 20 N grip, spillover remains tightly bounded."""
    res_20 = simulate_corticospinal_digit_grasp(target_force_n=20.0)
    
    # Spillover should not exceed 1.5 N
    assert res_20["final_spillover_force_n"] < 1.5
    assert res_20["final_fractionation_index"] > 0.85
