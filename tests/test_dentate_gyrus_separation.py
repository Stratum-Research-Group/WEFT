"""
Tests for Dentate Gyrus Sparse Pattern Separation, Orthogonalization, and Lean 4 Invariants.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import pytest
import os
import subprocess
import json
from weft.kernels.fused_dentate_gyrus_sparse_triton import compute_sparse_dg_step

def test_dentate_gyrus_sparse_step_suprathreshold():
    # Strong input drive should exceed threshold
    x_ec = [1.0, 2.0, 1.5]
    weights_col = [1.0, 1.0, 1.0] # dot = 4.5
    out = compute_sparse_dg_step(x_ec, weights_col, v_thresh=2.5)
    assert out > 0.0
    assert round(out, 2) == 2.0

def test_dentate_gyrus_sparse_step_subthreshold():
    # Weak input drive should be completely silenced
    x_ec = [0.2, 0.3, 0.1]
    weights_col = [1.0, 1.0, 1.0] # dot = 0.6 < 2.5
    out = compute_sparse_dg_step(x_ec, weights_col, v_thresh=2.5)
    assert out == 0.0

def test_dentate_gyrus_avx2_binary():
    bin_path = "build/weft-bench-dentate-gyrus"
    assert os.path.exists(bin_path), "Dentate gyrus binary missing"
    res = subprocess.run([bin_path, "1024", "100"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "WEFT AVX2 Dentate Gyrus Sparse Separation Microbenchmark" in res.stdout
    assert "Throughput:" in res.stdout
    assert "Sparsity Rate:" in res.stdout

def test_dentate_gyrus_separation_report():
    report_path = "build/dentate_gyrus_separation_report.json"
    assert os.path.exists(report_path)
    with open(report_path) as f:
        data = json.load(f)
    assert data["pattern_separation_verified"] is True
    assert data["ec_input_overlap"] > 0.75
    assert data["dg_output_overlap"] <= 0.15
    assert data["decorrelation_ratio"] >= 4.5
