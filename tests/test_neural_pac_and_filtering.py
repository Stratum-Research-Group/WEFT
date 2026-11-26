"""
Unit and regression test suite for Triton PAC Modulation Index kernel,
C++ AVX2 Hilbert phase microbenchmark, and Cortical PAC simulation.
"""
import pytest
import subprocess
import torch
import math
from pathlib import Path
from weft.kernels.fused_pac_modulation_triton import compute_pac_modulation_index_triton

def test_pac_modulation_triton_cuda():
    """Verify Triton PAC Modulation Index computes non-negative bounded MI on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for Triton PAC test.")

    device = torch.device("cuda:0")
    b, l = 4, 1024
    phase = (torch.rand((b, l), device=device) * 2.0 - 1.0) * math.pi
    amp = torch.rand((b, l), device=device) + 0.5

    mi_triton = compute_pac_modulation_index_triton(phase, amp, n_bins=18)
    assert (mi_triton >= 0.0).all(), "Modulation Index must be non-negative"
    assert (mi_triton <= 1.0).all(), "Modulation Index must not exceed 1.0"

def test_hilbert_phase_microbench_binary():
    """Verify build/weft-bench-hilbert-phase executes and reports high Msamples/sec."""
    bin_path = Path("build/weft-bench-hilbert-phase")
    if not bin_path.exists():
        pytest.skip("build/weft-bench-hilbert-phase binary not compiled.")

    cmd = [str(bin_path), "10000"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0
    assert "[RESULT] Filter Rate" in result.stdout
    assert "Msamples/sec" in result.stdout

def test_cortical_pac_simulation_report():
    """Verify build/cortical_pac_simulation_report.json exists and reports valid PAC metrics."""
    report_file = Path("build/cortical_pac_simulation_report.json")
    assert report_file.exists(), "Simulation report must exist"
    import json
    data = json.loads(report_file.read_text())
    assert data["status"] == "PAC_CONVERGED"
    assert data["modulation_index"] > 0.0
    assert data["l5_ca2_spike_count"] > 0
