"""
Unit and regression test suite for Triton RoPE kernel and native BPE Trie tokenizer.
Validates GPU L2 norm preservation and native C++ tokenizer benchmark execution.
"""
import pytest
import subprocess
import os
import torch
from pathlib import Path
from weft.kernels.fused_rope_triton import fused_rope_triton

def test_rope_gpu_norm_preservation():
    """Verify that Triton RoPE kernel preserves exact L2 vector norms within float16 tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for Triton RoPE tests.")
    
    device = torch.device("cuda:0")
    b, h, s, d = 2, 4, 64, 64
    x = torch.randn((b, h, s, d), dtype=torch.float16, device=device)
    
    # Compute baseline per-head per-token norms
    norms_in = torch.linalg.vector_norm(x.float(), dim=-1)
    
    # Apply fused RoPE
    out = fused_rope_triton(x, base_freq=10000.0)
    
    # Compute output per-head per-token norms
    norms_out = torch.linalg.vector_norm(out.float(), dim=-1)
    
    # Check max absolute drift (float16 quantization noise in trig operations gives ~1.5e-3)
    max_diff = torch.max(torch.abs(norms_in - norms_out)).item()
    assert max_diff < 2e-3, f"RoPE rotated norm violated: max difference = {max_diff}"

def test_weft_bench_tokenizer_binary():
    """Verify that build/weft-bench-tokenizer executes and reports throughput."""
    bin_path = Path("build/weft-bench-tokenizer")
    if not bin_path.exists():
        pytest.skip("build/weft-bench-tokenizer binary not compiled.")
    
    cmd = [str(bin_path), "10000"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0, f"Benchmark tool failed: {result.stderr}"
    assert "[RESULT] Throughput" in result.stdout
    assert "Mtok/s" in result.stdout
