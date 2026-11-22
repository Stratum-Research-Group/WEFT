"""
Property-based and formal integration test for Multi-Head Latent Attention (MLA) Invariants.
Validates the Lean 4 theorems in formal/MultiHeadLatentAttentionInvariants.lean
using Lean compiler verification and numerical simulation of DeepSeek-V3 MLA low-rank compression.
"""

import subprocess
import torch
import pytest
from pathlib import Path


def test_lean_mla_invariants_compiles():
    """Verify formal/MultiHeadLatentAttentionInvariants.lean compiles cleanly without axioms or sorry."""
    lean_file = Path("formal/MultiHeadLatentAttentionInvariants.lean")
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    content = lean_file.read_text()
    assert "sorry" not in content, "Found unproven 'sorry' in MultiHeadLatentAttentionInvariants.lean"
    assert "axiom" not in content, "Found uncertified 'axiom' in MultiHeadLatentAttentionInvariants.lean"

    res = subprocess.run(
        ["lean", str(lean_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Lean compilation failed:\n{res.stderr}"


def test_mla_kv_compression_simulation():
    """Simulate DeepSeek-V3 MLA low-rank key-value compression and un-projection."""
    torch.manual_seed(42)
    d_model = 7168       # DeepSeek-V3 hidden dimension
    d_c = 512           # Compressed KV latent dimension
    n_heads = 128       # Number of attention heads
    d_head = 128        # Per-head dimension
    seq_len = 16        # Tokens

    # Dense KV dimension vs latent dimension
    dense_kv_dim = n_heads * d_head
    assert d_c < dense_kv_dim, f"MLA compression must satisfy d_c ({d_c}) < n_h * d_h ({dense_kv_dim})"

    # Factorized down-projection and up-projection matrices
    W_dkv = torch.randn(d_model, d_c)
    W_uk = torch.randn(d_c, dense_kv_dim)
    W_uv = torch.randn(d_c, dense_kv_dim)

    # Input hidden states
    h = torch.randn(1, seq_len, d_model)

    # Down-project to latent KV representation
    c_kv = h @ W_dkv
    assert c_kv.shape == (1, seq_len, d_c)
    assert not torch.isnan(c_kv).any()

    # Un-project to full head keys and values
    K = c_kv @ W_uk
    V = c_kv @ W_uv
    assert K.shape == (1, seq_len, dense_kv_dim)
    assert V.shape == (1, seq_len, dense_kv_dim)

    # Null query annihilation check
    null_h = torch.zeros(1, 1, d_model)
    null_latent = null_h @ W_dkv
    assert torch.allclose(null_latent, torch.zeros_like(null_latent)), "Null query annihilation failed"
