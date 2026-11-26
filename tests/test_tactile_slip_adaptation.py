import pytest
import numpy as np
from scripts.sim_tactile_slip_compensation import simulate_tactile_slip_compensation

def test_tactile_slip_arrest_and_escalation():
    """Verify that an unexpected load perturbation triggers FA-I/FA-II bursts and arrests slip within 2.5 mm."""
    res = simulate_tactile_slip_compensation(
        baseline_grip_n=12.0,
        baseline_load_n=3.0,
        slip_load_perturbation_n=3.5,
        friction_coefficient_mu=0.35
    )
    
    # 1. Slip must be detected and subsequently arrested
    assert res["slip_start_ms"] is not None
    assert res["slip_arrest_ms"] is not None
    assert res["slip_duration_ms"] < 120.0, f"Slip took too long to arrest: {res['slip_duration_ms']} ms"
    
    # 2. Total displacement before arrest bounded < 2.5 mm
    assert res["total_slip_displacement_mm"] < 2.5, f"Displacement exceeded threshold: {res['total_slip_displacement_mm']} mm"
    
    # 3. Mechanoreceptor burst rates during slip exceed resting rates
    assert res["peak_fa1_rate_hz"] > 200.0, f"FA-I rate insufficient: {res['peak_fa1_rate_hz']} Hz"
    assert res["peak_fa2_rate_hz"] > 300.0, f"FA-II rate insufficient: {res['peak_fa2_rate_hz']} Hz"
    
    # 4. Final safety margin ratio must be >= 1.25x
    assert res["final_safety_margin_ratio"] >= 1.25, f"Final safety margin too low: {res['final_safety_margin_ratio']}"
    
    # 5. Grip force must escalate significantly above baseline (12 N -> > 20 N)
    assert res["final_adapted_grip_n"] > 20.0, f"Grip force did not escalate sufficiently: {res['final_adapted_grip_n']} N"

def test_stable_hold_no_slip():
    """Verify that when load remains well within friction limits, no slip occurs and grip does not escalate."""
    res_stable = simulate_tactile_slip_compensation(
        baseline_grip_n=20.0,
        baseline_load_n=2.0,
        slip_load_perturbation_n=1.0, # Total load 3.0 N, friction limit = 0.35 * 20 = 7.0 N
        friction_coefficient_mu=0.35
    )
    
    assert res_stable["slip_start_ms"] is None
    assert res_stable["total_slip_displacement_mm"] == 0.0
    assert res_stable["peak_fa1_rate_hz"] == 15.0 # Baseline rate
    assert res_stable["peak_fa2_rate_hz"] == 25.0 # Baseline rate
    assert abs(res_stable["final_adapted_grip_n"] - 20.0) < 0.1

def test_friction_coefficient_influence():
    """Verify that lower friction surfaces require higher adapted grip force."""
    res_slippery = simulate_tactile_slip_compensation(
        baseline_grip_n=12.0,
        baseline_load_n=3.0,
        slip_load_perturbation_n=3.0,
        friction_coefficient_mu=0.25 # More slippery
    )
    res_rough = simulate_tactile_slip_compensation(
        baseline_grip_n=12.0,
        baseline_load_n=3.0,
        slip_load_perturbation_n=3.0,
        friction_coefficient_mu=0.50 # High friction
    )
    
    assert res_slippery["final_adapted_grip_n"] > res_rough["final_adapted_grip_n"]
