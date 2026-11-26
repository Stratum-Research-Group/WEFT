"""
Unit and regression tests for Target #406: Fibrinogen-SHED-CM Molecular Interactions
Validates amyloidogenic clot formation, secretome shielding factor, Mac-1 suppression, and synaptic preservation.
"""

import pytest
import numpy as np

def compute_fibrin_clot_dynamics(fib, thr, abeta, shed_dose):
    f_part = fib / 3.0
    t_part = thr / 1.5
    a_part = 1.0 + abeta / 50.0
    raw_clot = min(100.0, f_part * t_part * a_part * 35.0)

    shielding = min(0.92, (shed_dose / 100.0) * 0.95)
    eff_clot = raw_clot * (1.0 - shielding)
    mac1 = eff_clot * 0.92 * (1.0 - (shed_dose / 100.0) * 0.35)
    spine_loss = mac1 * 0.82

    return raw_clot, eff_clot, mac1, spine_loss

def test_untreated_extravasation_triggers_massive_clot_and_spine_loss():
    raw_clot, eff_clot, mac1, spine_loss = compute_fibrin_clot_dynamics(
        fib=3.5, thr=2.0, abeta=90.0, shed_dose=0.0
    )
    assert raw_clot == 100.0
    assert eff_clot == 100.0
    assert mac1 >= 90.0, f"Untreated Mac-1 {mac1} must be >= 90%"
    assert spine_loss >= 70.0, f"Untreated spine loss {spine_loss} must be >= 70%"

def test_shed_cm_infusion_shields_fibrin_and_rescues_spines():
    raw_clot, eff_clot, mac1, spine_loss = compute_fibrin_clot_dynamics(
        fib=3.5, thr=2.0, abeta=90.0, shed_dose=100.0
    )
    assert eff_clot <= 10.0, f"Rescued clot {eff_clot} must be <= 10%"
    assert mac1 <= 10.0, f"Rescued Mac-1 {mac1} must be <= 10%"
    assert spine_loss <= 5.0, f"Rescued spine loss {spine_loss} must be <= 5%"

def test_physiological_intact_bbb_baseline():
    raw_clot, eff_clot, mac1, spine_loss = compute_fibrin_clot_dynamics(
        fib=0.1, thr=0.05, abeta=5.0, shed_dose=0.0
    )
    assert raw_clot < 1.0
    assert eff_clot < 1.0
    assert mac1 < 1.0
    assert spine_loss < 1.0

def test_dose_dependent_shielding_monotonicity():
    doses = [0.0, 25.0, 50.0, 75.0, 100.0]
    clots = [compute_fibrin_clot_dynamics(3.2, 1.85, 85.0, d)[1] for d in doses]
    # Check strict monotonic decrease
    for i in range(len(clots) - 1):
        assert clots[i] > clots[i+1], f"Clot at dose {doses[i]} ({clots[i]}) must be > than at {doses[i+1]} ({clots[i+1]})"
