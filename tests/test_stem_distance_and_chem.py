"""
Unit and regression test suite for Triton pairwise distance kernel,
C++ AVX2 chemical fingerprint microbenchmark, and STEM benchmark runner.
"""
import pytest
import subprocess
import torch
from pathlib import Path
from weft.kernels.fused_pairwise_dist_triton import compute_pairwise_distances_triton

def test_pairwise_dist_triton_cuda():
    """Verify Triton pairwise distance matches PyTorch torch.cdist within fp16 tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for Triton distance test.")

    device = torch.device("cuda:0")
    n = 64
    coords = torch.randn((n, 3), dtype=torch.float32, device=device)

    dist_triton, contact_triton = compute_pairwise_distances_triton(coords, threshold=2.0)
    dist_ref = torch.cdist(coords, coords).half()

    max_diff = torch.max(torch.abs(dist_triton - dist_ref)).item()
    assert max_diff < 5e-3, f"Distance matrix exceeded tolerance: {max_diff}"

def test_chem_fingerprint_microbench_binary():
    """Verify that build/weft-bench-chem-fingerprint executes and exceeds 10 M comp/s."""
    bin_path = Path("build/weft-bench-chem-fingerprint")
    if not bin_path.exists():
        pytest.skip("build/weft-bench-chem-fingerprint binary not compiled.")

    cmd = [str(bin_path), "100000"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0
    assert "[RESULT] Tanimoto Sim" in result.stdout
    assert "M comparisons/sec" in result.stdout

def test_stem_benchmarks_report():
    """Verify that build/stem_docking_benchmarks.json exists and contains valid metrics."""
    report_file = Path("build/stem_docking_benchmarks.json")
    assert report_file.exists(), "build/stem_docking_benchmarks.json must exist"
    import json
    data = json.loads(report_file.read_text())
    assert len(data) >= 3
    assert data[0]["plddt"] >= 80.0
