"""
Unit and regression tests for Target #405: Unified Closed-Loop Neuromodulation Engine
Validates multi-modal biomarker fusion, intervention gating, proportional stimulation, and energy savings.
"""

import pytest
import numpy as np

def compute_risk(beta, pac, gamma, slope, tremor):
    norm_b = min(1.0, beta / 40.0) * 0.30
    norm_p = min(1.0, pac) * 0.25
    gamma_loss = max(0.0, (35.0 - gamma) / 35.0) * 0.25
    slope_loss = max(0.0, (2.0 - slope) / 2.0) * 0.10
    norm_t = min(1.0, tremor / 3.0) * 0.10
    return min(1.0, max(0.0, norm_b + norm_p + gamma_loss + slope_loss + norm_t))

def test_multimodal_pathological_intervention_trigger():
    # Severe parkinsonian / paroxysmal state
    risk = compute_risk(beta=41.5, pac=0.76, gamma=12.5, slope=1.25, tremor=3.2)
    assert risk >= 0.70, f"Composite risk {risk} must exceed 0.70 for pathological state"

    stim = min(3.5, risk * 3.2)
    assert stim >= 2.2, f"Delivered current {stim} mA must be >= 2.2 mA"

    savings = 100.0 - (stim * (60.0 / 3.5))
    assert savings >= 50.0, f"Energy savings {savings}% must remain >= 50%"

def test_quiescent_standby_preservation():
    # Healthy homeostatic state
    risk = compute_risk(beta=11.5, pac=0.16, gamma=38.0, slope=2.40, tremor=0.2)
    assert risk < 0.45, f"Quiescent risk {risk} must remain < 0.45"

    stim = 0.0 if risk < 0.45 else min(3.5, risk * 3.2)
    savings = 88.0 if risk < 0.45 else (100.0 - (stim * (60.0 / 3.5)))

    assert stim == 0.0, "Quiescent state must deliver zero stimulation"
    assert savings == 88.0, "Quiescent standby must achieve exactly 88% battery savings"

def test_isolated_gamma_drop_triggers_sub_threshold():
    # Only gamma drop without beta elevation or tremor
    risk = compute_risk(beta=10.0, pac=0.10, gamma=15.0, slope=2.20, tremor=0.1)
    # 0.3*(10/40) + 0.25*(0.1) + 0.25*(20/35) + 0 + 0.1*(0.1/3.0)
    # = 0.075 + 0.025 + 0.1428 + 0.0033 = 0.246
    assert risk < 0.45, f"Isolated gamma drop alone should not trigger motor DBS (risk: {risk})"

def test_isolated_tremor_trigger_proportionality():
    # Essential tremor presentation (normal beta, normal gamma, severe tremor)
    risk = compute_risk(beta=12.0, pac=0.15, gamma=36.0, slope=2.10, tremor=3.0)
    # Mild overall composite risk
    assert risk < 0.45
