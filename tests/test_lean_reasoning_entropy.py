"""
Property-based and formal integration test for Reasoning Trace Entropy Bounds.
Validates the Lean 4 theorems in formal/ReasoningEntropyBounds.lean
using Lean compiler verification and numerical simulation on vocabulary logits.
"""

import subprocess
import torch
import pytest
from pathlib import Path


def test_lean_reasoning_entropy_compiles():
    """Verify formal/ReasoningEntropyBounds.lean compiles cleanly without axioms or sorry."""
    lean_file = Path("formal/ReasoningEntropyBounds.lean")
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    content = lean_file.read_text()
    assert "sorry" not in content, "Found unproven 'sorry' in ReasoningEntropyBounds.lean"
    assert "axiom" not in content, "Found uncertified 'axiom' in ReasoningEntropyBounds.lean"

    res = subprocess.run(
        ["lean", str(lean_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Lean compilation failed:\n{res.stderr}"


def test_reasoning_entropy_stability_simulation():
    """Simulate entropy drift under sub-bit low-rank perturbation on vocabulary logits."""
    torch.manual_seed(42)
    vocab_size = 152064  # DeepSeek / Qwen vocabulary size
    batch_size = 8

    # Realistic unperturbed logits with sharp reasoning probability distribution
    logits = torch.randn(batch_size, vocab_size) * 2.5
    probs = torch.softmax(logits, dim=-1)
    base_entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

    # Sub-bit quantization noise bounded by epsilon
    eps = 0.05
    perturbation = (torch.rand_like(logits) - 0.5) * 2 * eps
    quantized_logits = logits + perturbation
    quantized_probs = torch.softmax(quantized_logits, dim=-1)
    quantized_entropy = -torch.sum(quantized_probs * torch.log(quantized_probs + 1e-12), dim=-1)

    # Theoretical Bound: |H(P) - H(P_hat)| <= eps * log(vocab_size)
    max_drift = torch.max(torch.abs(base_entropy - quantized_entropy)).item()
    log_vocab = torch.log(torch.tensor(float(vocab_size))).item()
    theoretical_bound = eps * log_vocab

    assert max_drift <= theoretical_bound, (
        f"Empirical entropy drift {max_drift:.4f} exceeded theoretical bound {theoretical_bound:.4f}"
    )
    # Ensure policy does not collapse to zero
    assert (quantized_entropy > 0.1).all(), "Reasoning entropy collapsed to near zero!"
