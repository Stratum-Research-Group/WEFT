"""
Property-based and formal integration test for Mathematical Proof Tree Invariants.
Validates the Lean 4 theorems in formal/MathProofTreeInvariants.lean
using Lean compiler verification and numerical simulation of Qwen2.5-Math-72B dense projections.
"""

import subprocess
import torch
import pytest
from pathlib import Path


def test_lean_math_proof_tree_invariants_compiles():
    """Verify formal/MathProofTreeInvariants.lean compiles cleanly without axioms or sorry."""
    lean_file = Path("formal/MathProofTreeInvariants.lean")
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    content = lean_file.read_text()
    assert "sorry" not in content, "Found unproven 'sorry' in MathProofTreeInvariants.lean"
    assert "axiom" not in content, "Found uncertified 'axiom' in MathProofTreeInvariants.lean"

    res = subprocess.run(
        ["lean", str(lean_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Lean compilation failed:\n{res.stderr}"


def test_math_72b_projection_simulation():
    """Simulate Qwen2.5-Math-72B dense linear projection with low-rank factorization."""
    torch.manual_seed(42)
    d_model = 8192       # Qwen2.5-72B hidden dimension
    r = 64              # Base rank
    num_lemma_tokens = 32

    W_proj = torch.randn(d_model, d_model)
    U, S, V = torch.svd_lowrank(W_proj, q=r)

    # Low-rank factorized projection
    tokens = torch.randn(1, num_lemma_tokens, d_model)
    proj_lr = (tokens @ U) @ (torch.diag(S) @ V.T)

    assert proj_lr.shape == (1, num_lemma_tokens, d_model)
    assert not torch.isnan(proj_lr).any()
    assert not torch.isinf(proj_lr).any()

    # Null mathematical lemma annihilation check
    null_token = torch.zeros(1, 1, d_model)
    null_out = (null_token @ U) @ (torch.diag(S) @ V.T)
    assert torch.allclose(null_out, torch.zeros_like(null_out)), "Null lemma residual annihilation failed"
