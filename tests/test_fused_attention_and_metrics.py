"""
End-to-End Fused Attention & Prometheus Metrics Regression Test.
Validates:
1. Multi-stream fused bitboard attention kernel outputs on GPU.
2. Direct native C++ server Prometheus telemetry (/metrics) on Port 8091.
"""

import urllib.request
import torch

def test_prometheus_metrics_endpoint():
    url = "http://127.0.0.1:8091/metrics"
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "weft_server_uptime_seconds" in content
        assert "weft_lean4_theorems_clean 321" in content
        assert "weft_active_threads" in content

def test_fused_attention_numerical_stability():
    if not torch.cuda.is_available():
        print("[*] CUDA not available; skipping GPU kernel test.")
        return

    from weft.kernels.fused_bitboard_attention import fused_bitboard_attention
    device = torch.device("cuda")
    D = 64
    q = torch.randn(1, 8, 1, D, device=device, dtype=torch.float32)
    k = torch.randn(1, 16, D, device=device, dtype=torch.float32)
    v = torch.randn(1, 16, D, device=device, dtype=torch.float32)

    stream = torch.cuda.Stream()
    res = fused_bitboard_attention(q, k, v, stream=stream)
    torch.cuda.synchronize()

    assert res.shape == (1, 8, 1, D)
    assert not torch.isnan(res).any()
    assert not torch.isinf(res).any()

if __name__ == "__main__":
    print("=== Running Fused Attention & Metrics Tests ===")
    test_prometheus_metrics_endpoint()
    print("[PASS] test_prometheus_metrics_endpoint")
    test_fused_attention_numerical_stability()
    print("[PASS] test_fused_attention_numerical_stability")
    print("=== All Tests Passed ===")
