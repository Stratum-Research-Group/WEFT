"""
Unit tests for the Multi-Bit Continuum (0.18 to 1.58 BPW) engine.
"""

import torch
import pytest
from weft.multibit_continuum import HybridMultiBitLinear, analyze_pareto_frontier


def test_hybrid_multibit_linear_lifecycle():
    d_in, d_out, rank = 128, 256, 32
    layer = HybridMultiBitLinear(in_features=d_in, out_features=d_out, rank=rank, use_ternary_residual=True)
    
    x = torch.randn(2, 8, d_in)
    out = layer(x)
    
    assert out.shape == (2, 8, d_out)
    assert not torch.isnan(out).any()
    
    bpw = layer.effective_bpw()
    # Expect 2-bit residual + sub-bit base
    assert 2.0 < bpw < 5.0, f"Unexpected effective BPW: {bpw}"


def test_analyze_pareto_frontier():
    W = torch.randn(128, 64)
    results = analyze_pareto_frontier(W, rank_options=[16, 32])
    
    assert "frontier" in results
    assert len(results["frontier"]) == 4
    for point in results["frontier"]:
        assert "bpw" in point
        assert "relative_l2_error" in point
        assert point["bpw"] > 0
