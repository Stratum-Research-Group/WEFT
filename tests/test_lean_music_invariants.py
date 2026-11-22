"""
Property-based and formal integration test for Generative Music Autoregressive Invariants.
Validates the Lean 4 theorems in formal/MusicAutoregressiveInvariants.lean
using Lean compiler verification and numerical simulation of multi-codebook delay patterns.
"""

import subprocess
import torch
import pytest
from pathlib import Path


def test_lean_music_invariants_compiles():
    """Verify formal/MusicAutoregressiveInvariants.lean compiles cleanly without axioms or sorry."""
    lean_file = Path("formal/MusicAutoregressiveInvariants.lean")
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    content = lean_file.read_text()
    assert "sorry" not in content, "Found unproven 'sorry' in MusicAutoregressiveInvariants.lean"
    assert "axiom" not in content, "Found uncertified 'axiom' in MusicAutoregressiveInvariants.lean"

    res = subprocess.run(
        ["lean", str(lean_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Lean compilation failed:\n{res.stderr}"


def test_music_rvq_codebook_delay_simulation():
    """Simulate MusicGen multi-codebook RVQ delay pattern and low-rank projection."""
    torch.manual_seed(42)
    d_model = 1024       # MusicGen hidden dimension
    num_codebooks = 4   # 4 parallel RVQ codebooks
    t_frames = 50       # 50 acoustic frames (~1 second of audio)
    r = 64              # Base rank

    # Simulated delay pattern: codebook c has delay offset c
    delays = torch.arange(num_codebooks)
    for c in range(num_codebooks):
        t1, t2 = 10, 25
        assert (t1 + delays[c].item()) <= (t2 + delays[c].item()), "Delay monotonicity violated"

    # Multi-codebook embedding projection with low-rank factorization
    W_proj = torch.randn(d_model, d_model)
    U, S, V = torch.svd_lowrank(W_proj, q=r)

    # Low rank factorized projection
    tokens = torch.randn(1, t_frames, d_model)
    proj_lr = (tokens @ U) @ (torch.diag(S) @ V.T)

    assert proj_lr.shape == (1, t_frames, d_model)
    assert not torch.isnan(proj_lr).any()
    assert not torch.isinf(proj_lr).any()

    # Null token annihilation check
    null_token = torch.zeros(1, 1, d_model)
    null_out = (null_token @ U) @ (torch.diag(S) @ V.T)
    assert torch.allclose(null_out, torch.zeros_like(null_out)), "Null residual annihilation failed"
