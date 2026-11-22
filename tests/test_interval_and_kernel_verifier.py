"""
Tests for LeanKernelRuleVerifier and RigorousIntervalQuantizationValidator.
"""
import os
import shutil
import pytest
import torch

from weft.verifier import LeanKernelRuleVerifier, RewardBreakdown
from weft.interval_certifier import RigorousIntervalQuantizationValidator
from weft.format import WeftSubBitLinear


@pytest.mark.skipif(
    shutil.which("lean") is None and not os.path.exists(os.path.expanduser("~/.elan/bin/lean")),
    reason="Lean 4 compiler not installed"
)
def test_lean_kernel_rule_verifier():
    verifier = LeanKernelRuleVerifier()

    # Valid proof with zero axioms
    clean_proof = """
theorem add_comm_zero (n : Nat) : n + 0 = n := by
  rfl
"""
    res = verifier.verify_proof(
        theorem_name="add_comm_zero",
        theorem_statement="theorem add_comm_zero (n : Nat) : n + 0 = n",
        completion_code=clean_proof,
    )
    assert res.total_reward == 1.0
    assert res.details["status"] == "kernel_verified"

    # Incomplete proof using sorry
    sorry_proof = """
theorem add_comm_zero (n : Nat) : n + 0 = n := by
  sorry
"""
    res_sorry = verifier.verify_proof(
        theorem_name="add_comm_zero",
        theorem_statement="theorem add_comm_zero (n : Nat) : n + 0 = n",
        completion_code=sorry_proof,
    )
    assert res_sorry.total_reward == 0.0
    assert res_sorry.details["status"] == "verification_failed"


def test_rigorous_interval_quantization_validator():
    torch.manual_seed(42)
    m, n = 16, 16
    layer = WeftSubBitLinear(
        in_features=n,
        out_features=m,
        rank=4,
        num_paths=2,
    )
    W = torch.randn(m, n, dtype=torch.float32)

    res = RigorousIntervalQuantizationValidator.certify_reconstruction_error(W, layer)
    assert "rel_frobenius_interval" in res
    lo, hi = res["rel_frobenius_interval"]
    assert lo >= 0.0
    assert hi >= lo
    assert res["matrix_shape"] == "16x16"
    assert "certified_upper_bound" in res
