import pytest
import numpy as np
from scripts.sim_spinal_pad_gating import simulate_spinal_pad_gating

def test_spinal_pad_gating_during_reach():
    """Verify that voluntary reaching velocity engages PAD and suppresses reflex gain by > 80%."""
    res = simulate_spinal_pad_gating(peak_reach_velocity_m_s=1.8)
    
    # 1. PAD interneuron discharge must exceed 120 Hz during high-velocity reach
    assert res["peak_pad_rate_hz"] > 120.0, f"PAD interneuron rate too low: {res['peak_pad_rate_hz']} Hz"
    
    # 2. Terminal PAD depolarization must reach > +15 mV
    assert res["peak_terminal_depol_mv"] > 15.0, f"Insufficient depolarization: {res['peak_terminal_depol_mv']} mV"
    
    # 3. Peak reflex suppression must exceed 80%
    assert res["peak_suppression_pct"] > 80.0, f"Reflex suppression insufficient: {res['peak_suppression_pct']}%"
    
    # 4. Attenuated EPSP during reach must fall below 1.0 mV (vs > 3.0 mV baseline)
    assert res["minimum_epsp_mv"] < 1.0, f"Attenuated EPSP too high: {res['minimum_epsp_mv']} mV"
    assert res["baseline_epsp_mv"] > 3.0, f"Baseline EPSP too low: {res['baseline_epsp_mv']} mV"

def test_quiescent_reflex_preservation():
    """Verify that with zero voluntary drive, PAD remains at baseline and reflexes are unsuppressed."""
    res_rest = simulate_spinal_pad_gating(peak_reach_velocity_m_s=0.0)
    
    # Terminal depolarization should remain low (< 3 mV)
    assert res_rest["peak_terminal_depol_mv"] < 3.0
    # Suppression percentage should be minimal (< 35%)
    assert res_rest["peak_suppression_pct"] < 35.0
    # EPSP remains robust (> 2.5 mV)
    assert res_rest["minimum_epsp_mv"] > 2.5

def test_pad_velocity_monotonicity():
    """Verify that higher reaching velocity generates stronger PAD and greater reflex suppression."""
    res_low = simulate_spinal_pad_gating(peak_reach_velocity_m_s=0.8)
    res_high = simulate_spinal_pad_gating(peak_reach_velocity_m_s=2.2)
    
    assert res_high["peak_pad_rate_hz"] > res_low["peak_pad_rate_hz"]
    assert res_high["peak_terminal_depol_mv"] > res_low["peak_terminal_depol_mv"]
    assert res_high["peak_suppression_pct"] > res_low["peak_suppression_pct"]
    assert res_high["minimum_epsp_mv"] < res_low["minimum_epsp_mv"]
