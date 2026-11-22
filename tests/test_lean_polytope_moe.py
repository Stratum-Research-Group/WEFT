"""
Property-based and formal integration test for Polytope MoE Facet Invariants.
Validates the Lean 4 theorems in formal/PolytopeMoEInvariants.lean
using Lean compiler verification and numerical simulation.
"""

import subprocess
import torch
import pytest
from pathlib import Path


def test_lean_polytope_moe_proof_compiles():
    """Verify formal/PolytopeMoEInvariants.lean compiles cleanly without axioms or sorry."""
    lean_file = Path("formal/PolytopeMoEInvariants.lean")
    assert lean_file.exists(), f"Lean file not found: {lean_file}"
    
    # Check for unproven axioms or sorry placeholders
    content = lean_file.read_text()
    assert "sorry" not in content, "Found unproven 'sorry' in PolytopeMoEInvariants.lean"
    assert "axiom" not in content, "Found uncertified 'axiom' in PolytopeMoEInvariants.lean"

    res = subprocess.run(
        ["lean", str(lean_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Lean compilation failed:\n{res.stderr}"


def test_polytope_facet_margin_stability_simulation():
    """Simulate robust facet partitioning under sub-bit quantization perturbation."""
    torch.manual_seed(42)
    dim = 256
    num_experts = 8
    num_tokens = 64
    
    # Normalized token embeddings on unit sphere
    x = torch.randn(num_tokens, dim)
    x = x / torch.norm(x, dim=-1, keepdim=True)
    
    # Ideal continuous hyperplanes / centroids for each expert
    centroids = torch.randn(num_experts, dim)
    centroids = centroids / torch.norm(centroids, dim=-1, keepdim=True)
    
    # Cosine projection onto polytope expert normals
    logits = torch.matmul(x, centroids.T)  # [num_tokens, num_experts]
    ideal_assignments = torch.argmax(logits, dim=-1)
    
    # Quantized perturbation with operator norm bounded by delta
    delta = 0.05
    noise = torch.randn_like(centroids)
    noise = noise / torch.norm(noise, dim=-1, keepdim=True) * delta
    quantized_centroids = centroids + noise
    
    quantized_logits = torch.matmul(x, quantized_centroids.T)
    quantized_assignments = torch.argmax(quantized_logits, dim=-1)
    
    # Mathematical Bound:
    # |x . c_e - x . c'_e| <= ||x|| * ||c_e - c'_e|| <= 1 * delta = delta.
    # Therefore, between top-1 and any other expert e:
    # (x . c'_top1) >= (x . c_top1) - delta
    # (x . c'_e)    <= (x . c_e)    + delta
    # If (x . c_top1) - (x . c_top2) > 2 * delta, then
    # (x . c'_top1) - (x . c'_e) > 0 for all e != top1.
    sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
    margins = sorted_logits[:, 0] - sorted_logits[:, 1]
    
    # Theoretical stability margin condition: margin > 2 * delta
    robust_mask = margins > (2.0 * delta)
    assert robust_mask.sum() > 0, "Expected at least some tokens to satisfy robust margin"
    
    drift_on_robust = (ideal_assignments[robust_mask] != quantized_assignments[robust_mask]).sum()
    assert drift_on_robust == 0, f"Expected 0 drift on robust margin tokens, got {drift_on_robust}"
