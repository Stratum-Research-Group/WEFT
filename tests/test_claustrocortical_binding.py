"""
Tests for Claustrocortical Salience Kernel, Cross-Modal Perceptual Binding, and Lean 4 Invariants.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import pytest
import os
import subprocess
import json
from weft.kernels.fused_claustrocortical_salience_triton import compute_claustrocortical_salience_step

def test_claustrocortical_salience_amplification():
    # Salient channel (cue = 1.0) should be amplified
    stim = 10.0
    out_salient = compute_claustrocortical_salience_step(stim, cue=1.0, g_cl=0.8, gamma_cl=0.8)
    assert out_salient > stim
    assert round(out_salient, 2) == round(10.0 * (1.0 + 0.8 * 0.8), 2)

def test_claustrocortical_distractor_suppression():
    # Non-salient channel (cue = 0.0) should be suppressed
    stim = 10.0
    out_distractor = compute_claustrocortical_salience_step(stim, cue=0.0, g_cl=0.8, alpha_gaba=0.85)
    assert out_distractor < stim
    assert round(out_distractor, 2) == round(10.0 * (1.0 - 0.85 * 0.8), 2)

def test_claustrocortical_avx2_binary():
    bin_path = "build/weft-bench-claustrocortical-network"
    assert os.path.exists(bin_path), "Claustrocortical binary missing"
    res = subprocess.run([bin_path, "1024", "100"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "WEFT AVX2 Claustrocortical Salience Microbenchmark" in res.stdout
    assert "Throughput:" in res.stdout

def test_claustrocortical_binding_report():
    report_path = "build/claustrocortical_binding_report.json"
    assert os.path.exists(report_path)
    with open(report_path) as f:
        data = json.load(f)
    assert data["salience_binding_verified"] is True
    assert data["snr_enhancement_factor"] >= 5.0
    assert data["mean_ratio_claustrum_active"] > 6.0
