"""
Unit and regression tests for Target #409: Microglial Mac-1 & NLRP3 Inflammasome Gating
Validates fibrin gamma390-396 masking, Mac-1 ligation, ASC speck assembly, Caspase-1 cleavage, and M1/M2 resolution.
"""

import pytest
import numpy as np

def compute_inflammasome_state(gamma, tlr4, p2x7, shed_dose):
    shield = min(0.92, (shed_dose / 100.0) * 0.95)
    eff_gamma = gamma * (1.0 - shield)

    itgam_mod = 1.0 - (shed_dose / 100.0) * 0.65
    mac1 = max(2.0, min(100.0, eff_gamma * 1.05 * itgam_mod))

    raw_prime = tlr4 * 0.40 + mac1 * 0.60
    prime = raw_prime * (1.0 - (shed_dose / 100.0) * 0.70)
    act = p2x7 * (1.0 - (shed_dose / 100.0) * 0.50)

    speck = min(100.0, (prime * act) / 75.0)
    casp1 = speck * 0.94
    il1b = min(600.0, 15.0 + casp1 * 5.2)

    raw_ratio = (mac1 / 15.0) * (1.0 - (shed_dose / 100.0) * 0.88)
    ratio = max(0.12, min(6.0, raw_ratio))

    return mac1, speck, casp1, il1b, ratio

def test_unshielded_fibrin_triggers_severe_inflammasome():
    mac1, speck, casp1, il1b, ratio = compute_inflammasome_state(
        gamma=95.0, tlr4=85.0, p2x7=90.0, shed_dose=0.0
    )
    assert mac1 >= 85.0, f"Unshielded Mac-1 {mac1}% must be >= 85%"
    assert speck >= 85.0, f"Unshielded ASC specks {speck}% must be >= 85%"
    assert casp1 >= 80.0, f"Active Caspase-1 {casp1}% must be >= 80%"
    assert il1b >= 400.0, f"Mature IL-1beta {il1b} pg/mL must be >= 400"
    assert ratio >= 4.0, f"M1/M2 phenotype ratio {ratio}x must indicate neurotoxic polarization"

def test_shed_cm_infusion_quenches_inflammasome_and_drives_m2():
    mac1, speck, casp1, il1b, ratio = compute_inflammasome_state(
        gamma=95.0, tlr4=85.0, p2x7=90.0, shed_dose=100.0
    )
    assert mac1 <= 5.0, f"Rescued Mac-1 {mac1}% must be <= 5%"
    assert speck <= 15.0, f"Rescued ASC specks {speck}% must be <= 15%"
    assert il1b <= 90.0, f"Rescued IL-1beta {il1b} pg/mL must be <= 90"
    assert ratio <= 0.30, f"Rescued M1/M2 ratio {ratio}x must polarize into M2 resolution (<0.30x)"

def test_homeostatic_baseline():
    mac1, speck, casp1, il1b, ratio = compute_inflammasome_state(
        gamma=5.0, tlr4=10.0, p2x7=10.0, shed_dose=0.0
    )
    assert mac1 <= 10.0
    assert speck <= 5.0
    assert il1b <= 25.0
    assert ratio <= 0.50

def test_two_signal_requirement():
    # Signal 1 without Signal 2 (no ATP / P2X7)
    mac1, speck, casp1, il1b, ratio = compute_inflammasome_state(
        gamma=95.0, tlr4=85.0, p2x7=0.0, shed_dose=0.0
    )
    assert speck == 0.0, "Absence of Signal 2 (P2X7) must produce zero ASC specks"
    assert il1b == 15.0, "Absence of Caspase-1 must maintain baseline IL-1beta"
