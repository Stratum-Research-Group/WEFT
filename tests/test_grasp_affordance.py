import pytest
import numpy as np
from scripts.sim_grasp_affordance_preshaping import simulate_grasp_affordance_preshaping

def test_grasp_affordance_precision_pinch():
    """Verify that a 25 mm object selects precision pinch and matches Jeannerod 70% peak aperture timing."""
    res = simulate_grasp_affordance_preshaping(object_width_mm=25.0)
    
    # 1. Grip prototype must be precision pinch (< 35 mm)
    assert res["grip_mode"] == "PRECISION_PINCH", f"Unexpected mode: {res['grip_mode']}"
    
    # 2. Peak aperture timing must occur at 65 - 75% of reach duration
    assert 0.65 <= res["peak_time_fraction"] <= 0.75, f"Peak timing outside Jeannerod window: {res['peak_time_fraction']}"
    
    # 3. Peak aperture must strictly enclose object with safety margin (> 10 mm margin)
    assert res["overshoot_margin_mm"] > 10.0, f"Insufficient safety margin: {res['overshoot_margin_mm']} mm"
    assert res["measured_peak_aperture_mm"] > res["object_width_mm"]
    
    # 4. Final contact aperture should settle to object width + 0.5 mm
    assert abs(res["final_contact_aperture_mm"] - (res["object_width_mm"] + 0.5)) < 0.1

def test_grasp_affordance_power_grip():
    """Verify that a 50 mm object selects whole-hand power grasp."""
    res_power = simulate_grasp_affordance_preshaping(object_width_mm=50.0)
    
    assert res_power["grip_mode"] == "POWER_GRASP"
    assert res_power["measured_peak_aperture_mm"] > res_power["object_width_mm"]
    assert res_power["measured_peak_aperture_mm"] > 55.0

def test_peak_aperture_monotonicity():
    """Verify that larger objects produce larger peak hand preshaping apertures."""
    res_20 = simulate_grasp_affordance_preshaping(object_width_mm=20.0)
    res_40 = simulate_grasp_affordance_preshaping(object_width_mm=40.0)
    res_60 = simulate_grasp_affordance_preshaping(object_width_mm=60.0)
    
    assert res_20["measured_peak_aperture_mm"] < res_40["measured_peak_aperture_mm"]
    assert res_40["measured_peak_aperture_mm"] < res_60["measured_peak_aperture_mm"]
