"""
Tests for Hippocampal Phase Precession, SWR Sequence Replay, and Lean 4 Invariants.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import pytest
import os
import subprocess
import json
import math
from weft.kernels.fused_phase_precession_triton import compute_phase_precession_point

def test_phase_precession_point_in_field():
    # Center of field: u = 0.5, target phase ~ pi
    prob = compute_phase_precession_point(x=1.0, x_center=1.0, width=0.8, theta_phase=math.pi)
    assert prob > 0.5

def test_phase_precession_outside_field():
    # Outside field: x = 2.0 while field is [0.6, 1.4]
    prob = compute_phase_precession_point(x=2.0, x_center=1.0, width=0.8, theta_phase=math.pi)
    assert prob == 0.0

def test_hippocampal_precession_avx2_binary():
    bin_path = "build/weft-bench-hippocampal-precession"
    assert os.path.exists(bin_path), "Hippocampal precession binary missing"
    res = subprocess.run([bin_path, "1024", "100"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "WEFT AVX2 Hippocampal Phase Precession Microbenchmark" in res.stdout
    assert "Throughput:" in res.stdout

def test_hippocampal_replay_report():
    report_path = "build/hippocampal_replay_report.json"
    assert os.path.exists(report_path)
    with open(report_path) as f:
        data = json.load(f)
    assert data["replay_fidelity_verified"] is True
    assert data["order_preserved"] is True
    assert data["compression_ratio"] >= 15.0
    assert data["mean_swr_isi_ms"] < 20.0
