"""
Unit and regression test suite for Triton SwiGLU activation,
C++ AVX2 vector cosine similarity microbenchmark, and Sub-bit Model Convertor.
"""
import pytest
import subprocess
import torch
import torch.nn.functional as F
from pathlib import Path
from weft.kernels.fused_swiglu_triton import fused_swiglu_triton

def test_triton_swiglu_numerical_equivalence():
    """Verify that Triton SwiGLU matches PyTorch F.silu(gate) * up within fp16 tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for Triton SwiGLU test.")
    
    device = torch.device("cuda:0")
    m, n = 128, 2048
    gate = torch.randn((m, n), dtype=torch.float16, device=device)
    up = torch.randn((m, n), dtype=torch.float16, device=device)

    out_triton = fused_swiglu_triton(gate, up)
    out_ref = (F.silu(gate.float()) * up.float()).half()

    max_diff = torch.max(torch.abs(out_triton - out_ref)).item()
    assert max_diff < 5e-3, f"SwiGLU output difference exceeded tolerance: {max_diff}"

def test_weft_bench_vector_search_binary():
    """Verify that build/weft-bench-vector-search executes and reports high QPS throughput."""
    bin_path = Path("build/weft-bench-vector-search")
    if not bin_path.exists():
        pytest.skip("build/weft-bench-vector-search binary not compiled.")
    
    cmd = [str(bin_path), "1000"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0, f"Benchmark tool failed: {result.stderr}"
    assert "[RESULT] Cosine Result" in result.stdout
    assert "queries/sec" in result.stdout

def test_subbit_convertor_script():
    """Verify that scripts/convert_hf_to_subbit.py generates a valid .weft binary."""
    out_file = Path("build/test_pytest_convert.weft")
    if out_file.exists():
        out_file.unlink()
        
    cmd = [
        sys_py := ".venv/bin/python",
        "scripts/convert_hf_to_subbit.py",
        "--name", "pytest-model",
        "--out", str(out_file)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0
    assert out_file.exists()
    assert out_file.stat().st_size > 1024
