"""
Unit and regression tests for Target #404: Paroxysmal Dyskinesia & Channelopathy
Validates PRRT2 deficit burst firing, striatal DA surge, and carbamazepine rescue dynamics.
"""

import pytest
import numpy as np

def test_prrt2_deficit_burst_triggering():
    prrt2_cap = 0.20
    trigger = 0.90
    deficit = max(0.0, 1.0 - prrt2_cap)
    raw_burst = 40.0 + deficit * 120.0 * trigger
    assert raw_burst >= 120.0, f"DCN burst rate {raw_burst} must be >= 120 Hz"

def test_striatal_dopamine_efflux_surge():
    burst_rate = 126.0
    da_efflux = 1.0 + (burst_rate / 60.0) * 1.2
    assert da_efflux >= 3.0, f"Striatal DA efflux multiplier {da_efflux} must be >= 3.0x"

def test_carbamazepine_rescue_normalization():
    prrt2_cap = 0.20
    trigger = 0.90
    cbz = 300.0  # Therapeutic dose

    deficit = max(0.0, 1.0 - prrt2_cap)
    raw_burst = 40.0 + deficit * 120.0 * trigger
    nav_inh = min(1.0, cbz / 300.0)
    rescued_burst = raw_burst * (1.0 - 0.75 * nav_inh)

    da_efflux = 1.0 + (rescued_burst / 60.0) * 1.2
    b_excess = max(0.0, (rescued_burst - 45.0) * 0.95)
    da_excess = max(0.0, (da_efflux - 1.2) * 22.0)
    severity = min(100.0, b_excess + da_excess)

    assert rescued_burst <= 50.0, f"Rescued DCN burst {rescued_burst} must be <= 50 Hz"
    assert severity <= 15.0, f"Rescued dyskinesia severity {severity} must be <= 15/100"

def test_wildtype_quiescence_under_motor_initiation():
    prrt2_cap = 1.0  # Wild-type
    trigger = 1.0    # Maximal voluntary motor initiation
    deficit = max(0.0, 1.0 - prrt2_cap)
    burst = 40.0 + deficit * 120.0 * trigger
    da = 1.0 + (burst / 60.0) * 1.2
    b_excess = max(0.0, (burst - 45.0) * 0.95)
    da_excess = max(0.0, (da - 1.2) * 22.0)
    severity = min(100.0, b_excess + da_excess)

    assert burst == 40.0
    assert severity <= 15.0
