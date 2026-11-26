"""
Tests for Corticothalamic Layer VI Feedback, Receptive Field Sharpening, and Lean 4 Invariants.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import pytest
import os
import subprocess
import json
from weft.kernels.fused_corticothalamic_feedback_triton import compute_corticothalamic_feedback_step

def test_corticothalamic_feedback_center_facilitation():
    # Layer VI feedback (l6 = 1.0) should amplify center input
    c_in = 10.0
    s_in = 10.0
    c_out, s_out = compute_corticothalamic_feedback_step(c_in, s_in, l6=1.0, gamma_l6=0.45)
    assert c_out > c_in
    assert round(c_out, 2) == 14.5

def test_corticothalamic_feedback_surround_suppression():
    # Layer VI feedback should suppress surround input via TRN
    c_in = 10.0
    s_in = 10.0
    c_out, s_out = compute_corticothalamic_feedback_step(c_in, s_in, l6=1.0, alpha_trn=0.80)
    assert s_out < s_in
    assert round(s_out, 2) == 2.0

def test_corticothalamic_avx2_binary():
    bin_path = "build/weft-bench-corticothalamic-feedback"
    assert os.path.exists(bin_path), "Corticothalamic binary missing"
    res = subprocess.run([bin_path, "1024", "100"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "WEFT AVX2 Corticothalamic Layer VI Feedback Microbenchmark" in res.stdout
    assert "Throughput:" in res.stdout

def test_corticothalamic_rf_report():
    report_path = "build/corticothalamic_rf_report.json"
    assert os.path.exists(report_path)
    with open(report_path) as f:
        data = json.load(f)
    assert data["receptive_field_sharpening_verified"] is True
    assert data["sharpening_factor"] >= 3.5
    assert data["mean_contrast_feedback"] > 5.0
