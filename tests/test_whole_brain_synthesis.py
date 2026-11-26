"""
Tests for Whole-Brain Macro-Connectome Integration, Cognitive Cycle Simulation, and Lean 4 Invariants.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import pytest
import os
import subprocess
import json
from weft.kernels.fused_macro_connectome_triton import compute_macro_connectome_step

def test_macro_connectome_step_integration():
    # Co-active drives should integrate and remain bounded within v_max
    out = compute_macro_connectome_step(
        ctx=1.0, thal=2.0, cl=1.5, bg=2.0, cer=1.0, v_max=10.0
    )
    assert out > 1.0
    assert out <= 10.0

def test_macro_connectome_step_saturation():
    # Extreme inputs should clamp to v_max
    out_saturated = compute_macro_connectome_step(
        ctx=100.0, thal=100.0, cl=100.0, bg=100.0, cer=100.0, v_max=10.0
    )
    assert out_saturated == 10.0

def test_whole_brain_avx2_binary():
    bin_path = "build/weft-bench-whole-brain"
    assert os.path.exists(bin_path), "Whole brain binary missing"
    res = subprocess.run([bin_path, "1024", "100"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "WEFT AVX2 Whole-Brain Macro-Circuit Microbenchmark" in res.stdout
    assert "Throughput:" in res.stdout

def test_whole_brain_synthesis_report():
    report_path = "build/whole_brain_synthesis_report.json"
    assert os.path.exists(report_path)
    with open(report_path) as f:
        data = json.load(f)
    assert data["macro_circuit_closed_loop_verified"] is True
    assert data["success_rate_pct"] == 100.0
    assert data["mean_latency_ms"] < 20.0
