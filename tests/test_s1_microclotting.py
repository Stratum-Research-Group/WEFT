"""
Unit and regression tests for Target #408: SARS-CoV-2 S1 Spike-Induced Vascular Microclotting
Validates S1 spike-induced amyloid microclot drive, ULVWF multimer index, tissue StO2 hypoxia, and SHED-CM rescue.
"""

import pytest
import numpy as np

def compute_s1_microclot_dynamics(s1, fib, net, adamts, shed_dose):
    s_part = s1 / 20.0
    f_part = fib / 3.0
    n_part = 1.0 + net / 100.0
    raw_clot = min(100.0, s_part * f_part * n_part * 25.0)

    shield = min(0.92, (shed_dose / 100.0) * 0.95)
    eff_clot = raw_clot * (1.0 - shield)

    safe_a = max(15.0, adamts)
    raw_u = 1.0 + (s1 / 15.0) * (100.0 / safe_a) * 0.4
    eff_u = max(1.0, raw_u * (1.0 - (shed_dose / 100.0) * 0.65))

    sto2 = max(65.0, min(98.0, 96.0 - eff_clot * 0.32))

    return raw_clot, eff_clot, eff_u, sto2

def test_s1_spike_challenge_induces_severe_thrombosis_and_hypoxia():
    raw_clot, eff_clot, eff_u, sto2 = compute_s1_microclot_dynamics(
        s1=45.0, fib=4.2, net=84.0, adamts=25.0, shed_dose=0.0
    )
    assert raw_clot == 100.0
    assert eff_clot == 100.0
    assert eff_u >= 3.0, f"Injured ULVWF multimer index {eff_u} must be >= 3.0x"
    assert sto2 <= 65.0, f"Tissue StO2 {sto2}% must drop to severe hypoxia floor"

def test_shed_cm_infusion_clears_microclots_and_restores_perfusion():
    raw_clot, eff_clot, eff_u, sto2 = compute_s1_microclot_dynamics(
        s1=45.0, fib=4.2, net=15.0, adamts=85.0, shed_dose=100.0
    )
    assert eff_clot <= 10.0, f"Rescued microclot density {eff_clot}% must be <= 10%"
    assert eff_u <= 1.5, f"Rescued ULVWF index {eff_u} must be <= 1.5x"
    assert sto2 >= 93.0, f"Rescued capillary StO2 {sto2}% must exceed 93%"

def test_physiological_baseline_without_s1_spike():
    raw_clot, eff_clot, eff_u, sto2 = compute_s1_microclot_dynamics(
        s1=0.0, fib=3.0, net=10.0, adamts=90.0, shed_dose=0.0
    )
    assert raw_clot == 0.0
    assert eff_clot == 0.0
    assert eff_u == 1.0
    assert sto2 == 96.0

def test_s1_titer_dose_response_monotonicity():
    titers = [0.0, 15.0, 30.0, 45.0, 60.0]
    clots = [compute_s1_microclot_dynamics(t, 3.5, 50.0, 50.0, 0.0)[1] for t in titers]
    for i in range(len(clots) - 1):
        assert clots[i] <= clots[i+1]
