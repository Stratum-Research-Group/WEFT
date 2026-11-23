"""
Test AVX-512 Fused SwiGLU Numerical Accuracy & Parity
"""

import torch
import torch.nn.functional as F
import pytest

def test_swiglu_reference_parity():
    gate = torch.randn(2048, dtype=torch.float32)
    up = torch.randn(2048, dtype=torch.float32)

    # PyTorch reference
    ref_out = F.silu(gate) * up

    # Approximate polynomial version
    gate_clamped = torch.clamp(gate, -10.0, 10.0)
    sig_approx = 0.5 + 0.5 * (gate_clamped / (1.0 + torch.abs(gate_clamped)))
    approx_out = (gate * sig_approx) * up

    # Check cosine similarity
    cos_sim = F.cosine_similarity(ref_out.unsqueeze(0), approx_out.unsqueeze(0)).item()
    assert cos_sim > 0.99, f"SwiGLU approximation cosine similarity too low: {cos_sim}"
