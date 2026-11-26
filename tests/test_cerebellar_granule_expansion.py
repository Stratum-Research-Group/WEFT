"""
End-to-end tests for Cerebellar Granule Layer Sparse Expansion & Golgi Feedback Inhibition.
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import torch
import numpy as np

from weft.kernels.fused_cerebellar_granule_expansion_triton import cerebellar_granule_expansion
from scripts.sim_cerebellar_granule_expansion import run_cerebellar_simulation


def test_cerebellar_granule_expansion_shapes_and_execution():
    """Verify that Cerebellar Granule 4-claw expansion kernel executes properly and outputs valid shapes."""
    B, N_MF, N_GRC = 2, 64, 6400
    torch.manual_seed(42)

    x_mf = torch.relu(torch.randn(B, N_MF))
    claws_idx = torch.randint(0, N_MF, (N_GRC, 4))
    claws_w = torch.rand(N_GRC, 4) * 0.4 + 0.2
    i_golgi = torch.tensor([0.6, 0.6])

    y_grc, active_frac = cerebellar_granule_expansion(
        x_mf, claws_idx, claws_w, i_golgi, theta_grc=0.5
    )

    assert y_grc.shape == (B, N_GRC)
    assert 0.0 <= active_frac <= 1.0
    assert (y_grc >= 0.0).all()


def test_cerebellar_granule_sparsity_enforcement():
    """Verify that strong Golgi inhibition reduces active granule fraction below 1.5%."""
    B, N_MF, N_GRC = 1, 64, 3200
    x_mf = torch.ones(B, N_MF) * 0.5
    claws_idx = torch.randint(0, N_MF, (N_GRC, 4))
    claws_w = torch.ones(N_GRC, 4) * 0.4

    # High Golgi inhibition
    i_golgi_high = torch.tensor([1.2])
    _, active_frac_sparse = cerebellar_granule_expansion(
        x_mf, claws_idx, claws_w, i_golgi_high, theta_grc=0.5
    )

    assert active_frac_sparse < 0.015


def test_cerebellar_expansion_decorrelation():
    """Verify that divergent expansion recoding orthogonalizes overlapping input cues."""
    report = run_cerebellar_simulation(num_mf=64, num_grc=6400, seed=42)
    assert report["output_jaccard_overlap"] < report["input_jaccard_overlap"]
    assert report["decorrelation_ratio"] > 1.5
    assert report["enforces_ultra_sparsity"] is True
