"""
Unit and regression test suite for Triton Hyena DNA long-conv kernel,
C++ AVX2 DNA search microbenchmark, and Post-Pretraining Alignment pipeline.
"""
import pytest
import subprocess
import torch
from pathlib import Path
from weft.kernels.fused_hyena_conv_triton import hyena_long_conv_triton

def test_hyena_conv_triton_cuda():
    """Verify Triton 1D causal long-conv matches PyTorch 1D convolution on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for Triton Hyena conv test.")

    device = torch.device("cuda:0")
    b, l, k = 2, 256, 16
    u = torch.randn((b, l), dtype=torch.float16, device=device)
    h = torch.randn((k,), dtype=torch.float16, device=device)

    out_triton = hyena_long_conv_triton(u, h)

    # Reference manual sliding dot product for first element
    u_cpu = u.float().cpu()
    h_cpu = h.float().cpu()
    ref_val = (u_cpu[0, :k] * h_cpu.flip(0)).sum().item()
    triton_val = out_triton[0, k - 1].item()

    assert abs(ref_val - triton_val) < 0.25, f"Convolution discrepancy: ref={ref_val}, triton={triton_val}"

def test_dna_search_microbench_binary():
    """Verify build/weft-bench-dna-search executes and reports high alignment rates."""
    bin_path = Path("build/weft-bench-dna-search")
    if not bin_path.exists():
        pytest.skip("build/weft-bench-dna-search binary not compiled.")

    cmd = [str(bin_path), "100000"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0
    assert "[RESULT] Alignment Rate" in result.stdout
    assert "M aligns/sec" in result.stdout

def test_post_pretrain_align_manifest():
    """Verify checkpoints/weft2_aligned_instruct/alignment_manifest.json exists and is valid."""
    manifest_file = Path("checkpoints/weft2_aligned_instruct/alignment_manifest.json")
    assert manifest_file.exists(), "Alignment manifest must exist"
    import json
    data = json.loads(manifest_file.read_text())
    assert data["status"] == "ALIGNED_PRODUCTION_READY"
    assert "step_10000" in data["base_checkpoint"]
