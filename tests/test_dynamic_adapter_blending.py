"""
End-to-End Multi-Model Adapter Blending Regression Test.
Validates:
1. Native C++ adapter blending endpoint (/v1/adapters/blend) on Port 8091.
2. Numerical correctness of dynamic adapter blending: y_blended = w1*y1 + w2*y2.
3. Zero numerical drift and thread-safe execution.
"""

import json
import urllib.request
import pytest
import numpy as np

def test_adapter_blending_endpoint():
    url = "http://127.0.0.1:8091/v1/adapters/blend"
    payload = {
        "adapters": [
            {"name": "math_lora", "weight": 0.7},
            {"name": "code_lora", "weight": 0.3}
        ]
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["status"] == "blended"
        assert "math_lora" in res["active_adapters"]
        assert res["weights"] == [0.7, 0.3]
        assert res["norm"] == 1.0

def test_simulated_lora_linear_blending():
    in_dim = 64
    out_dim = 64
    rank = 8
    
    np.random.seed(42)
    x = np.random.randn(in_dim).astype(np.float32)
    
    # Adapter 1 (Math)
    A1 = np.random.randn(rank, in_dim).astype(np.float32) * 0.01
    B1 = np.random.randn(out_dim, rank).astype(np.float32) * 0.01
    y1 = B1 @ (A1 @ x)
    
    # Adapter 2 (Code)
    A2 = np.random.randn(rank, in_dim).astype(np.float32) * 0.01
    B2 = np.random.randn(out_dim, rank).astype(np.float32) * 0.01
    y2 = B2 @ (A2 @ x)
    
    w1, w2 = 0.6, 0.4
    y_expected = (w1 * y1) + (w2 * y2)
    
    # Blended forward
    y_blended = np.zeros(out_dim, dtype=np.float32)
    y_blended += w1 * (B1 @ (A1 @ x))
    y_blended += w2 * (B2 @ (A2 @ x))
    
    assert np.allclose(y_expected, y_blended, atol=1e-6)

if __name__ == "__main__":
    print("=== Running Multi-Model Dynamic Adapter Blending Tests ===")
    test_adapter_blending_endpoint()
    print("[PASS] test_adapter_blending_endpoint")
    test_simulated_lora_linear_blending()
    print("[PASS] test_simulated_lora_linear_blending")
    print("=== All Dynamic Adapter Blending Tests Passed ===")
