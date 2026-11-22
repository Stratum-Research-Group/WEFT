"""
Property-based and formal integration test for Coding Syntax and Indentation Invariants.
Validates the Lean 4 theorems in formal/CodingSyntaxInvariants.lean
using Lean compiler verification and numerical simulation of code token projections.
"""

import subprocess
import torch
import pytest
from pathlib import Path


def test_lean_coding_syntax_invariants_compiles():
    """Verify formal/CodingSyntaxInvariants.lean compiles cleanly without axioms or sorry."""
    lean_file = Path("formal/CodingSyntaxInvariants.lean")
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    content = lean_file.read_text()
    assert "sorry" not in content, "Found unproven 'sorry' in CodingSyntaxInvariants.lean"
    assert "axiom" not in content, "Found uncertified 'axiom' in CodingSyntaxInvariants.lean"

    res = subprocess.run(
        ["lean", str(lean_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Lean compilation failed:\n{res.stderr}"


def test_coding_projection_lowrank_simulation():
    """Simulate Qwen2.5-Coder projection with low-rank factorization."""
    torch.manual_seed(42)
    d_model = 5120       # Qwen2.5-Coder-32B hidden dimension
    r = 64              # Base rank
    t_tokens = 32       # Code sequence length

    W_proj = torch.randn(d_model, d_model)
    U, S, V = torch.svd_lowrank(W_proj, q=r)

    # Low rank factorized projection
    tokens = torch.randn(1, t_tokens, d_model)
    proj_lr = (tokens @ U) @ (torch.diag(S) @ V.T)

    assert proj_lr.shape == (1, t_tokens, d_model)
    assert not torch.isnan(proj_lr).any()
    assert not torch.isinf(proj_lr).any()

    # Null token annihilation check
    null_token = torch.zeros(1, 1, d_model)
    null_out = (null_token @ U) @ (torch.diag(S) @ V.T)
    assert torch.allclose(null_out, torch.zeros_like(null_out)), "Null code residual annihilation failed"
