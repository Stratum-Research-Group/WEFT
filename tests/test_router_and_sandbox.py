import pytest
import os
from weft.adapter_router import HierarchicalAdapterRouter
from weft.code_sandbox import EdgeCodeSandbox

def test_hierarchical_adapter_router():
    router = HierarchicalAdapterRouter()
    results = router.route_query_to_adapter("critical line zeros of riemann zeta", top_k=3)
    
    assert len(results) == 3
    assert results[0]["similarity_score"] > 0
    assert results[0]["retrieval_latency_us"] < 10000.0  # Under 10 ms across all 161 adapters

def test_edge_code_sandbox_python():
    sandbox = EdgeCodeSandbox()
    code = "x = 42\ny = 58\nprint(f'Sum: {x + y}')"
    res = sandbox.execute_python_code(code)
    
    assert res["success"] is True
    assert "Sum: 100" in res["stdout"]
    assert res["execution_time_ms"] < 50.0

def test_edge_code_sandbox_error_containment():
    sandbox = EdgeCodeSandbox()
    code = "1 / 0"
    res = sandbox.execute_python_code(code)
    
    assert res["success"] is False
    assert "ZeroDivisionError" in res["stderr"]
