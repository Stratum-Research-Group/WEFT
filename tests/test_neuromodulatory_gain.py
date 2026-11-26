"""
Tests for Tri-Monoaminergic Gain Modulation, Yerkes-Dodson Exploration, and Lean 4 Invariants.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import pytest
import os
import subprocess
import json
from weft.kernels.fused_neuromodulatory_gain_triton import compute_neuromodulatory_gain_step

def test_neuromodulatory_gain_ne_boost():
    # Elevating NE increases feedforward gain
    i_low_ne = compute_neuromodulatory_gain_step(i_ff=10.0, i_rec=0.0, ne=0.1, ach=0.5)
    i_high_ne = compute_neuromodulatory_gain_step(i_ff=10.0, i_rec=0.0, ne=0.9, ach=0.5)
    assert i_high_ne > i_low_ne

def test_neuromodulatory_gain_ach_snr():
    # Elevating ACh suppresses recurrent noise
    i_low_ach = compute_neuromodulatory_gain_step(i_ff=0.0, i_rec=10.0, ne=0.5, ach=0.1)
    i_high_ach = compute_neuromodulatory_gain_step(i_ff=0.0, i_rec=10.0, ne=0.5, ach=0.9)
    assert i_high_ach < i_low_ach

def test_neuromodulation_avx2_binary():
    bin_path = "build/weft-bench-neuromodulation"
    assert os.path.exists(bin_path), "Neuromodulation binary missing"
    res = subprocess.run([bin_path, "1024", "100"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "WEFT AVX2 Neuromodulatory Gain Microbenchmark" in res.stdout
    assert "Throughput:" in res.stdout

def test_yerkes_dodson_report():
    report_path = "build/yerkes_dodson_report.json"
    assert os.path.exists(report_path)
    with open(report_path) as f:
        data = json.load(f)
    assert data["inverted_u_transition_verified"] is True
    assert data["phase1_exploitation_accuracy_pct"] >= 90.0
    assert data["phase2_exploration_rate_pct"] >= 15.0
