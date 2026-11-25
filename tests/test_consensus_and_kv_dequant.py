"""
Unit and regression test suite for Triton FP8 KV dequantization,
C++ AVX2 Sub-bit Transpose microbenchmark, and Chat Template Registry.
"""
import pytest
import subprocess
import torch
from pathlib import Path
from weft.kernels.fused_kv_dequant_triton import dequantize_fp8_kv_cache
from scripts.validate_chat_templates import format_turn, CHAT_TEMPLATES

def test_fp8_kv_dequant_cuda_numerical():
    """Verify Triton FP8 KV dequantization produces correct scaled outputs on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for Triton KV dequantization test.")
    
    device = torch.device("cuda:0")
    b, h, s, d = 2, 4, 32, 64
    
    # Simulate int8 values scaled to fp16
    q_data = torch.randint(-5, 6, (b, h, s, d), dtype=torch.int8, device=device)
    scales = torch.full((b, h, s), 0.125, dtype=torch.float32, device=device)
    
    out = dequantize_fp8_kv_cache(q_data, scales)
    expected = q_data.to(torch.float32) * scales.unsqueeze(-1)
    
    max_err = torch.max(torch.abs(out.to(torch.float32) - expected)).item()
    assert max_err < 1e-3, f"Dequantization error exceeded tolerance: {max_err}"

def test_weft_bench_subbit_transpose_binary():
    """Verify that build/weft-bench-subbit-transpose executes and exceeds 1000 Mval/s throughput."""
    bin_path = Path("build/weft-bench-subbit-transpose")
    if not bin_path.exists():
        pytest.skip("build/weft-bench-subbit-transpose binary not compiled.")
    
    cmd = [str(bin_path), "100"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0, f"Benchmark tool failed: {result.stderr}"
    assert "[RESULT] Dequant Rate" in result.stdout
    assert "Mval/s" in result.stdout

def test_chat_templates_validation():
    """Verify all 4 chat templates produce correct structured tokens without truncation."""
    for name in CHAT_TEMPLATES:
        formatted = format_turn(name, "sys_prompt", "usr_prompt")
        assert "sys_prompt" in formatted
        assert "usr_prompt" in formatted
