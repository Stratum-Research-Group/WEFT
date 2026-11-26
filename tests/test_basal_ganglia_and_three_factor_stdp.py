"""
Tests for Basal Ganglia Attractor Selection, Three-Factor Dopaminergic STDP, and Lean 4 Invariants.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import pytest
import os
import subprocess
import json
from weft.kernels.fused_three_factor_stdp_triton import compute_three_factor_step

def test_three_factor_stdp_step():
    w_curr = 0.5
    e_curr = 0.8
    da_positive = 1.0
    eta = 0.05

    w_new, e_new = compute_three_factor_step(w_curr, e_curr, da_positive, eta=eta, trace_decay=0.9)
    assert w_new > w_curr
    assert round(w_new, 4) == round(0.5 + 0.05 * 1.0 * 0.8, 4)
    assert round(e_new, 4) == round(0.8 * 0.9, 4)

def test_three_factor_stdp_clamping():
    w_curr = 0.98
    e_curr = 1.0
    da_surge = 2.0
    eta = 0.1
    w_new, _ = compute_three_factor_step(w_curr, e_curr, da_surge, eta=eta, w_max=1.0)
    assert w_new == 1.0

def test_basal_ganglia_avx2_binary():
    bin_path = "build/weft-bench-basal-ganglia"
    assert os.path.exists(bin_path), "Basal ganglia binary missing"
    res = subprocess.run([bin_path, "1024", "100"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "WEFT AVX2 Basal Ganglia Attractor Selection Microbenchmark" in res.stdout
    assert "Throughput:" in res.stdout

def test_corticostriatal_switching_report():
    report_path = "build/corticostriatal_switching_report.json"
    assert os.path.exists(report_path)
    with open(report_path) as f:
        data = json.load(f)
    assert data["clean_switching_verified"] is True
    assert data["stn_hyperdirect_brake_freeze_steps"] > 10
    assert data["phase1_selected_attractor"] == 0
    assert data["phase2_selected_attractor"] == 1
