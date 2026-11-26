"""
Tests for Amygdalar Associative Plasticity, Prefrontal Extinction, and Lean 4 Invariants.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import pytest
import os
import subprocess
import json
from weft.kernels.fused_amygdalar_plasticity_triton import compute_amygdalar_step

def test_amygdalar_associative_ltp_step():
    # CS paired with US shock should induce LTP
    w_init = 0.3
    w_new, cea_out = compute_amygdalar_step(w_curr=w_init, cs=1.0, us=1.0, vmpfc=0.0, eta_ltp=0.1)
    assert w_new > w_init
    assert round(w_new, 2) == 0.40

def test_amygdalar_extinction_itc_clamp():
    # Strong vmPFC drive should silence CeA threat output
    w_pot = 0.8
    _, cea_out = compute_amygdalar_step(w_curr=w_pot, cs=1.0, us=0.0, vmpfc=1.0, alpha_itc=0.95)
    assert cea_out == 0.0 # 0.8 - 0.95 < 0 -> clamped to 0.0

def test_amygdalar_avx2_binary():
    bin_path = "build/weft-bench-amygdalar-circuit"
    assert os.path.exists(bin_path), "Amygdalar binary missing"
    res = subprocess.run([bin_path, "1024", "100"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "WEFT AVX2 Amygdalar Conditioning & Extinction Microbenchmark" in res.stdout
    assert "Throughput:" in res.stdout

def test_amygdalar_conditioning_report():
    report_path = "build/amygdalar_conditioning_report.json"
    assert os.path.exists(report_path)
    with open(report_path) as f:
        data = json.load(f)
    assert data["threat_and_extinction_verified"] is True
    assert data["mean_conditioned_cea"] > 0.6
    assert data["extinction_suppression_pct"] > 90.0
