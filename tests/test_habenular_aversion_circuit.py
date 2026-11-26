"""
Tests for Lateral Habenula Aversion Kernel, Dopaminergic Dip Dynamics, and Lean 4 Invariants.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import pytest
import os
import subprocess
import json
from weft.kernels.fused_lateral_habenula_triton import compute_habenular_step

def test_habenular_negative_rpe_dip():
    # Negative RPE (-0.8) should excite LHb and drop dopamine to zero
    lhb_act, da_out = compute_habenular_step(rpe=-0.8, aversive=0.2, da_base=0.5, beta_rpe=1.5, gamma_rmtg=0.4)
    assert lhb_act > 1.0
    assert da_out == 0.0 # 0.5 - 0.4 * (1.2 + 0.16) = 0.5 - 0.544 < 0 -> clamped to 0.0

def test_habenular_positive_rpe_burst():
    # Positive RPE (+0.8) should keep LHb low and elevate dopamine
    lhb_act, da_out = compute_habenular_step(rpe=0.8, aversive=0.0, da_base=0.5)
    assert lhb_act == 0.0
    assert da_out > 0.5

def test_habenular_avx2_binary():
    bin_path = "build/weft-bench-habenular-circuit"
    assert os.path.exists(bin_path), "Habenular binary missing"
    res = subprocess.run([bin_path, "1024", "100"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "WEFT AVX2 Lateral Habenula - RMTg Microbenchmark" in res.stdout
    assert "Throughput:" in res.stdout
    assert "Dopamine Dips:" in res.stdout

def test_habenular_aversion_report():
    report_path = "build/habenular_aversion_report.json"
    assert os.path.exists(report_path)
    with open(report_path) as f:
        data = json.load(f)
    assert data["habenular_aversion_verified"] is True
    assert data["da_suppression_pct"] > 80.0
    assert data["mean_lhb_burst"] > 1.0
