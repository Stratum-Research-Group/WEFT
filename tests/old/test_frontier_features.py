import pytest
from weft.unified_sandbox import UnifiedAgentSandbox
from weft.test_time_compute import TestTimeVerifier


def test_unified_sandbox_python_execution():
    sandbox = UnifiedAgentSandbox()

    code = """
def factorial(n):
    return 1 if n <= 1 else n * factorial(n - 1)
ans = factorial(5)
print(f"Result: {ans}")
"""
    res = sandbox.execute_python_code(code)
    assert res["status"] == "success"
    assert "Result: 120" in res["stdout"]
    assert sandbox.scope["ans"] == 120


def test_unified_sandbox_error_capture():
    sandbox = UnifiedAgentSandbox()
    res = sandbox.execute_python_code("1 / 0")
    assert res["status"] == "error"
    assert "ZeroDivisionError" in res["error"]


def test_test_time_verifier_entropy():
    import torch
    verifier = TestTimeVerifier(entropy_threshold=2.0)
    # Uniform logits = maximum entropy
    uniform_logits = torch.ones(10)
    ent_uniform = verifier.calculate_entropy(uniform_logits)

    # Peaked logits = low entropy
    peaked_logits = torch.tensor([10.0] + [0.0] * 9)
    ent_peaked = verifier.calculate_entropy(peaked_logits)

    assert ent_uniform > ent_peaked
