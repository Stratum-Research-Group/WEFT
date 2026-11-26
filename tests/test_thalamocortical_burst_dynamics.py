"""
Tests for Thalamocortical Burst Dynamics, T-Type Calcium Gating, and Lean 4 Invariants.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import pytest
import os
import subprocess
import json
from weft.kernels.fused_ttype_calcium_triton import compute_ttype_calcium_step

def test_ttype_calcium_step_deinactivation():
    # Hyperpolarized state (-85 mV): h should increase toward h_inf(0.73) (de-inactivation)
    h_init = 0.2
    h_new, i_t = compute_ttype_calcium_step(v=-85.0, h=h_init, dt=5.0)
    assert h_new > h_init
    assert i_t < 0.0 # Inward depolarizing calcium current

def test_ttype_calcium_step_inactivation():
    # Depolarized state (-50 mV): h should decrease toward 0 (inactivation)
    h_init = 0.8
    h_new, i_t = compute_ttype_calcium_step(v=-50.0, h=h_init, dt=5.0)
    assert h_new < h_init

def test_thalamic_relay_avx2_binary():
    bin_path = "build/weft-bench-thalamic-relay"
    assert os.path.exists(bin_path), "Thalamic relay binary missing"
    res = subprocess.run([bin_path, "1024", "100"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "WEFT AVX2 Thalamocortical Relay Microbenchmark" in res.stdout
    assert "Throughput:" in res.stdout
    assert "Burst Amplification Ratio:" in res.stdout

def test_thalamocortical_spindle_report():
    report_path = "build/thalamocortical_spindle_report.json"
    assert os.path.exists(report_path)
    with open(report_path) as f:
        data = json.load(f)
    assert data["in_spindle_range_7_to_14hz"] is True
    assert 7.0 <= data["spindle_frequency_hz"] <= 14.0
