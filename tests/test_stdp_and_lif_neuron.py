"""
Unit and integration tests for STDP kernels, Lean formal invariants, and LIF microbenchmark.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import pytest
import math
import os
import subprocess
from weft.kernels.fused_stdp_event_triton import compute_stdp_delta, apply_stdp_event_driven

def test_stdp_delta_causal_and_anticausal():
    # Causal pairing: pre before post (dt > 0)
    dw_causal = compute_stdp_delta(dt=5.0, a_plus=0.1, tau_plus=20.0)
    assert dw_causal > 0.0
    expected_causal = 0.1 * math.exp(-5.0 / 20.0)
    assert math.isclose(dw_causal, expected_causal, rel_tol=1e-5)

    # Anti-causal pairing: post before pre (dt < 0)
    dw_anticausal = compute_stdp_delta(dt=-5.0, a_minus=0.12, tau_minus=20.0)
    assert dw_anticausal < 0.0
    expected_anticausal = -0.12 * math.exp(-5.0 / 20.0)
    assert math.isclose(dw_anticausal, expected_anticausal, rel_tol=1e-5)

    # Simultaneous pairing (dt = 0)
    dw_zero = compute_stdp_delta(dt=0.0)
    assert dw_zero == 0.0

def test_stdp_bounded_clamping():
    w_min = 0.0
    w_max = 1.0

    # Potentiation clamped at w_max
    w_curr = 0.95
    w_pot = apply_stdp_event_driven(w_curr, dt=1.0, a_plus=0.2, w_min=w_min, w_max=w_max)
    assert w_pot == 1.0

    # Depression clamped at w_min
    w_curr = 0.05
    w_dep = apply_stdp_event_driven(w_curr, dt=-1.0, a_minus=0.2, w_min=w_min, w_max=w_max)
    assert w_dep == 0.0

def test_lif_neuron_avx2_binary():
    bin_path = "build/weft-bench-lif-neuron"
    assert os.path.exists(bin_path), "Microbenchmark binary missing"
    res = subprocess.run([bin_path, "1024", "100"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "WEFT AVX2 LIF Neuron Microbenchmark" in res.stdout
    assert "Throughput:" in res.stdout

def test_associative_memory_report():
    report_path = "build/stdp_associative_memory_report.json"
    assert os.path.exists(report_path)
    import json
    with open(report_path) as f:
        data = json.load(f)
    assert data["recall_successful"] is True
    assert data["signal_to_noise_ratio_db"] > 3.0
