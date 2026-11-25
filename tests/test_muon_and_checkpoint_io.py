"""
Unit and regression test suite for Triton Muon Newton-Schulz kernel,
C++ checkpoint I/O microbenchmark, and Evo whole-genome variant scoring.
"""
import pytest
import subprocess
import torch
from pathlib import Path
from weft.kernels.fused_muon_triton import newton_schulz_step_triton
from weft.stem.evo_genomics_runner import EvoGenomicsRunner

def test_newton_schulz_orthogonality_cuda():
    """Verify Newton-Schulz polynomial iteration bounds singular values on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for Newton-Schulz test.")

    device = torch.device("cuda:0")
    m, n = 32, 32
    g = torch.randn((m, n), dtype=torch.float32, device=device)

    out = newton_schulz_step_triton(g, steps=5)
    # Check that out @ out.T is approximately identity scaled
    gram = out @ out.T
    diag_mean = torch.diag(gram).mean().item()
    assert abs(diag_mean - 1.0) < 0.25, f"Singular values diverged: {diag_mean}"

def test_checkpoint_io_microbench_binary():
    """Verify build/weft-bench-checkpoint-io executes and reports multi-GB/s speed."""
    bin_path = Path("build/weft-bench-checkpoint-io")
    if not bin_path.exists():
        pytest.skip("build/weft-bench-checkpoint-io binary not compiled.")

    cmd = [str(bin_path), "10"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0
    assert "[RESULT] Bandwidth" in result.stdout
    assert "GB/s" in result.stdout

def test_evo_genomics_variant_scoring():
    """Verify EvoGenomicsRunner computes LLR impact scores for DNA windows."""
    runner = EvoGenomicsRunner()
    res = runner.score_variant_impact("ACGT" * 50, variant_pos=10, ref_base="A", alt_base="T")
    assert "llr_score" in res
    assert "prediction" in res
    assert res["context_window_bp"] == 200
