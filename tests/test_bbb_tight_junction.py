"""
Unit and regression tests for Target #407: Blood-Brain Barrier (BBB) Tight-Junction Restoration
Validates Tie2 receptor activation, MMP suppression, Claudin-5/Occludin reconstitution, and TEER recovery.
"""

import pytest
import numpy as np

def compute_bbb_state(peri, ang1, ros, mmp, shed_dose):
    raw_t = 20.0 + peri * 0.3 + ang1 * 3.2
    ros_factor = 1.0 - (ros / 200.0)
    eff_t = min(100.0, max(10.0, raw_t * ros_factor))

    m_supp = max(0.2, 1.0 - (shed_dose / 100.0) * 0.75)
    eff_m = mmp * m_supp

    claudin = min(100.0, max(5.0, eff_t * 1.1 - eff_m * 12.0))
    occludin = min(100.0, max(5.0, eff_t * 1.05 - eff_m * 10.5))

    teer = 200.0 + claudin * 8.5 + occludin * 4.5
    leakage = max(2.0, min(100.0, 100.0 - teer * (100.0 / 1500.0)))

    return eff_t, claudin, occludin, teer, leakage

def test_severe_injury_causes_bbb_collapse():
    eff_t, claudin, occludin, teer, leakage = compute_bbb_state(
        peri=22.0, ang1=0.0, ros=95.0, mmp=4.2, shed_dose=0.0
    )
    assert eff_t <= 20.0, f"Injured Tie2 {eff_t} must be <= 20%"
    assert claudin == 5.0, f"Degraded Claudin-5 {claudin} must reach floor"
    assert teer <= 350.0, f"Injured TEER {teer} must be <= 350 Ohm*cm^2"
    assert leakage >= 75.0, f"Paracellular leakage {leakage} must be >= 75%"

def test_shed_cm_infusion_fully_restores_bbb():
    eff_t, claudin, occludin, teer, leakage = compute_bbb_state(
        peri=85.0, ang1=18.0, ros=20.0, mmp=1.5, shed_dose=100.0
    )
    assert eff_t >= 90.0, f"Rescued Tie2 {eff_t} must be >= 90%"
    assert claudin >= 95.0, f"Rescued Claudin-5 {claudin} must be >= 95%"
    assert teer >= 1400.0, f"Rescued TEER {teer} must exceed 1400 Ohm*cm^2"
    assert leakage <= 5.0, f"Rescued leakage {leakage} must be <= 5%"

def test_pericyte_dose_dependency():
    # As pericyte coverage drops without treatment, TEER drops monotonically
    coverages = [85.0, 65.0, 45.0, 25.0]
    teers = [compute_bbb_state(c, 0.5, 20.0, 1.5, 0.0)[3] for c in coverages]
    for i in range(len(teers) - 1):
        assert teers[i] > teers[i+1], f"TEER at coverage {coverages[i]} must exceed that at {coverages[i+1]}"

def test_claudin_occludin_bounds():
    eff_t, claudin, occludin, teer, leakage = compute_bbb_state(
        peri=100.0, ang1=25.0, ros=0.0, mmp=0.5, shed_dose=100.0
    )
    assert 5.0 <= claudin <= 100.0
    assert 5.0 <= occludin <= 100.0
    assert 2.0 <= leakage <= 100.0
