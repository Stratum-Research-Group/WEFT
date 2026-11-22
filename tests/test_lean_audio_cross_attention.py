"""
Property-based and formal integration test for Audio Cross-Attention Invariants.
Validates the Lean 4 theorems in formal/AudioCrossAttentionInvariants.lean
using Lean compiler verification and numerical simulation of Whisper cross-attention.
"""

import subprocess
import torch
import pytest
from pathlib import Path


def test_lean_audio_cross_attention_compiles():
    """Verify formal/AudioCrossAttentionInvariants.lean compiles cleanly without axioms or sorry."""
    lean_file = Path("formal/AudioCrossAttentionInvariants.lean")
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    content = lean_file.read_text()
    assert "sorry" not in content, "Found unproven 'sorry' in AudioCrossAttentionInvariants.lean"
    assert "axiom" not in content, "Found uncertified 'axiom' in AudioCrossAttentionInvariants.lean"

    res = subprocess.run(
        ["lean", str(lean_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Lean compilation failed:\n{res.stderr}"


def test_audio_cross_attention_lowrank_simulation():
    """Simulate audio encoder-decoder cross-attention projection with low-rank factorization."""
    torch.manual_seed(42)
    d_model = 1280       # Whisper Large V3 hidden dimension
    r = 64              # Base rank
    t_text = 16         # Text tokens
    t_audio = 1500      # 30 seconds at 50Hz (subsampled from 3000 frames)

    # Dense projection weights for cross-attention Q and K
    W_q = torch.randn(d_model, d_model)
    W_k = torch.randn(d_model, d_model)

    # Low-rank factorizations
    U_q, S_q, V_q = torch.svd_lowrank(W_q, q=r)
    U_k, S_k, V_k = torch.svd_lowrank(W_k, q=r)

    # Verify storage compression bound
    dense_storage = 2 * (d_model * d_model)
    lowrank_storage = 2 * (2 * d_model * r + r)
    assert lowrank_storage < dense_storage, f"Low-rank storage {lowrank_storage} must be < dense {dense_storage}"

    # Forward cross-attention query-key dot-product
    h_text = torch.randn(1, t_text, d_model)
    h_audio = torch.randn(1, t_audio, d_model)

    Q_lr = (h_text @ U_q) @ (torch.diag(S_q) @ V_q.T)
    K_lr = (h_audio @ U_k) @ (torch.diag(S_k) @ V_k.T)

    attn_scores = torch.bmm(Q_lr, K_lr.transpose(1, 2)) / (d_model ** 0.5)
    assert attn_scores.shape == (1, t_text, t_audio)
    assert not torch.isnan(attn_scores).any(), "NaN found in low-rank cross-attention scores"
    assert not torch.isinf(attn_scores).any(), "Inf found in low-rank cross-attention scores"
