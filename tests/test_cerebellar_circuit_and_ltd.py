"""
Tests for Cerebellar LTD Kernel, Forward Model Sensorimotor Adaptation, and Lean 4 Invariants.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import pytest
import os
import subprocess
import json
from weft.kernels.fused_cerebellar_ltd_triton import compute_cerebellar_ltd_step

def test_cerebellar_ltd_step_depression():
    # CF error present: PF synapse should undergo LTD (decrease)
    w_curr = 0.8
    w_new = compute_cerebellar_ltd_step(w_curr, pf=1.0, cf=1.0, eta_ltd=0.05)
    assert w_new < w_curr
    assert round(w_new, 4) == 0.75

def test_cerebellar_ltd_step_recovery():
    # CF error absent: slow LTP recovery
    w_curr = 0.5
    w_new = compute_cerebellar_ltd_step(w_curr, pf=1.0, cf=0.0, eta_ltp=0.01)
    assert w_new > w_curr
    assert round(w_new, 4) == 0.51

def test_cerebellar_avx2_binary():
    bin_path = "build/weft-bench-cerebellar-circuit"
    assert os.path.exists(bin_path), "Cerebellar binary missing"
    res = subprocess.run([bin_path, "1024", "100"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "WEFT AVX2 Cerebellar Purkinje-DCN Microbenchmark" in res.stdout
    assert "Throughput:" in res.stdout
    assert "DCN Disinhibitions:" in res.stdout

def test_cerebellar_adaptation_report():
    report_path = "build/cerebellar_adaptation_report.json"
    assert os.path.exists(report_path)
    with open(report_path) as f:
        data = json.load(f)
    assert data["adaptation_successful"] is True
    assert data["error_reduction_pct"] > 80.0
    assert data["final_error_deg"] < 1.0
