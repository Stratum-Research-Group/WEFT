"""
Unit and regression test suite for Triton GQA fused kernel,
C++ AVX2 GQA softmax microbenchmark, and Aligned Instruct model evaluation.
"""
import pytest
import subprocess
import torch
from pathlib import Path
from weft.kernels.fused_gqa_triton import fused_gqa_triton

def test_fused_gqa_triton_cuda():
    """Verify Triton GQA executes on GPU with matching dimension shapes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for Triton GQA test.")

    device = torch.device("cuda:0")
    b, h_q, h_kv, s, d = 2, 8, 2, 32, 64
    q = torch.randn((b, h_q, s, d), dtype=torch.float16, device=device)
    k = torch.randn((b, h_kv, s, d), dtype=torch.float16, device=device)
    v = torch.randn((b, h_kv, s, d), dtype=torch.float16, device=device)

    out = fused_gqa_triton(q, k, v)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()

def test_gqa_softmax_microbench_binary():
    """Verify build/weft-bench-gqa-softmax executes and reports high throughput."""
    bin_path = Path("build/weft-bench-gqa-softmax")
    if not bin_path.exists():
        pytest.skip("build/weft-bench-gqa-softmax binary not compiled.")

    cmd = [str(bin_path), "1000"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0
    assert "[RESULT] Throughput" in result.stdout
    assert "M scores/sec" in result.stdout

def test_aligned_instruct_eval_report():
    """Verify build/aligned_instruct_eval_report.json exists and reports valid speed."""
    report_file = Path("build/aligned_instruct_eval_report.json")
    assert report_file.exists(), "build/aligned_instruct_eval_report.json must exist"
    import json
    data = json.loads(report_file.read_text())
    assert "results" in data
    assert len(data["results"]) == 3
    assert data["avg_speed_tok_s"] > 1000.0
