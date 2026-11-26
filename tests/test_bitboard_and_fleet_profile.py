"""
Unit and regression test suite for Triton Bitboard Mask Attention kernel,
C++ AVX2 Bitboard popcount microbenchmark, and Fleet Profiler report.
"""
import pytest
import subprocess
import torch
from pathlib import Path
from weft.kernels.fused_bitboard_mask_triton import fused_bitboard_mask_attention_triton

def test_bitboard_attention_triton_cuda():
    """Verify Triton bitboard causal mask attention executes on GPU with zero active tokens where bit=0."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for Triton bitboard test.")

    device = torch.device("cuda:0")
    b, h, s, d = 2, 4, 64, 64
    q = torch.randn((b, h, s, d), dtype=torch.float16, device=device)
    k = torch.randn((b, h, s, d), dtype=torch.float16, device=device)
    v = torch.randn((b, h, s, d), dtype=torch.float16, device=device)

    # Word with only lower 32 bits set
    bitboard = torch.tensor([[0x00000000FFFFFFFF], [0x00000000FFFFFFFF]], dtype=torch.int64, device=device)

    out = fused_bitboard_mask_attention_triton(q, k, v, bitboard)
    assert out.shape == q.shape

    # Check that upper 32 tokens are masked to zero
    upper_tokens_norm = torch.linalg.norm(out[:, :, 32:, :].float()).item()
    assert upper_tokens_norm == 0.0, "Inactive bitboard tokens must produce exact zero output"

def test_bitboard_popcount_microbench_binary():
    """Verify build/weft-bench-bitboard-popcount executes and reports high check rate."""
    bin_path = Path("build/weft-bench-bitboard-popcount")
    if not bin_path.exists():
        pytest.skip("build/weft-bench-bitboard-popcount binary not compiled.")

    cmd = [str(bin_path), "1000"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0
    assert "[RESULT] Popcount Rate" in result.stdout
    assert "M words/sec" in result.stdout

def test_fleet_throughput_profile_report():
    """Verify build/fleet_throughput_profile.json exists and contains 3 model records."""
    report_file = Path("build/fleet_throughput_profile.json")
    assert report_file.exists(), "Fleet profile report must exist"
    import json
    data = json.loads(report_file.read_text())
    assert len(data) == 3
    assert data[0]["decode_tok_s"] > 100.0
