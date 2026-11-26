"""
Unit and regression tests for Target #410: Glymphatic Convective Flux & AQP4 Polarization
Validates AQP4 polarization recovery, NREM delta volume expansion, Darcy convective flux, and solute clearance.
"""

import pytest
import numpy as np

def compute_glymphatic_state(pol, delta, vaso, shed_dose):
    eff_pol = min(100.0, max(15.0, pol + shed_dose * 0.55))
    dose_mod = 1.0 + shed_dose * (0.20 / 100.0)
    eff_alpha = min(25.0, max(12.0, 14.0 + (delta * 10.0) * dose_mod))

    v_norm = vaso / 3.5
    p_norm = eff_pol / 85.0
    a_norm = eff_alpha / 22.0
    flux = max(0.1, min(3.5, v_norm * p_norm * a_norm * 2.1))

    clearance = min(100.0, max(5.0, (flux / 2.0) * 75.0))
    retention = max(0.0, 100.0 - clearance)

    return eff_pol, eff_alpha, flux, clearance, retention

def test_glymphatic_stagnation_causes_severe_solute_retention():
    eff_pol, eff_alpha, flux, clearance, retention = compute_glymphatic_state(
        pol=25.0, delta=0.18, vaso=0.9, shed_dose=0.0
    )
    assert eff_pol == 25.0, "Injured AQP4 polarization must remain at 25%"
    assert flux <= 0.50, f"Stagnant flux {flux} um/min must be <= 0.50"
    assert clearance <= 25.0, f"Stagnant clearance {clearance}%/hr must be <= 25%"
    assert retention >= 75.0, f"Toxic solute retention {retention}% must be >= 75%"

def test_shed_cm_infusion_fully_reconstitutes_clearance():
    eff_pol, eff_alpha, flux, clearance, retention = compute_glymphatic_state(
        pol=35.0, delta=0.85, vaso=3.8, shed_dose=100.0
    )
    assert eff_pol >= 85.0, f"Rescued AQP4 polarization {eff_pol}% must be >= 85%"
    assert eff_alpha >= 22.0, f"Rescued extracellular volume {eff_alpha}% must be >= 22%"
    assert flux >= 2.0, f"Rescued convective flux {flux} um/min must exceed 2.0"
    assert clearance >= 90.0, f"Rescued clearance {clearance}%/hr must be >= 90%"
    assert retention <= 10.0, f"Rescued toxic retention {retention}% must be <= 10%"

def test_physiological_sleep_baseline():
    eff_pol, eff_alpha, flux, clearance, retention = compute_glymphatic_state(
        pol=85.0, delta=0.80, vaso=3.5, shed_dose=0.0
    )
    assert flux >= 1.8, f"Baseline sleep flux {flux} must be >= 1.8 um/min"
    assert clearance >= 70.0, f"Baseline clearance {clearance}% must be >= 70%"
    assert retention <= 30.0

def test_vasomotion_dose_dependency():
    vasomotions = [1.0, 2.0, 3.0, 4.0]
    fluxes = [compute_glymphatic_state(85.0, 0.80, v, 0.0)[2] for v in vasomotions]
    for i in range(len(fluxes) - 1):
        assert fluxes[i] < fluxes[i+1]
