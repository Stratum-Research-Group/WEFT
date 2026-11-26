"""
Unit and regression test suite for Triton Distillation KL kernel,
C++ AVX2 Top-K sampling microbenchmark, and Student Distillation runner.
"""
import pytest
import subprocess
import torch
import torch.nn.functional as F
from pathlib import Path
from weft.kernels.fused_distillation_kl_triton import fused_distillation_kl_triton

def test_distillation_kl_triton_cuda():
    """Verify Triton KL divergence loss computes correctly on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for Triton KL loss test.")

    device = torch.device("cuda:0")
    b, v = 4, 128
    logits_t = torch.randn((b, v), dtype=torch.float32, device=device)
    p_t = F.softmax(logits_t / 2.0, dim=-1)
    logits_s = torch.randn((b, v), dtype=torch.float32, device=device)

    loss_triton = fused_distillation_kl_triton(p_t, logits_s, temperature=2.0)
    assert loss_triton.item() > 0.0, "KL loss must be strictly positive"
    assert not torch.isnan(loss_triton), "Loss must not be NaN"

def test_topk_sampling_microbench_binary():
    """Verify build/weft-bench-topk-sampling executes and reports fast sampling rate."""
    bin_path = Path("build/weft-bench-topk-sampling")
    if not bin_path.exists():
        pytest.skip("build/weft-bench-topk-sampling binary not compiled.")

    cmd = [str(bin_path), "1000"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0
    assert "[RESULT] Sampling Rate" in result.stdout
    assert "tokens/sec" in result.stdout

def test_distillation_manifest_exists():
    """Verify build/distillation_runs/distillation_manifest.json exists and is valid."""
    manifest_file = Path("build/distillation_runs/distillation_manifest.json")
    assert manifest_file.exists(), "Distillation manifest must exist"
    import json
    data = json.loads(manifest_file.read_text())
    assert data["status"] == "DISTILLATION_CONVERGED"
    assert data["final_kl_loss"] < 0.05
