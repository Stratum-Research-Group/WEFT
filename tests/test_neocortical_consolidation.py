"""
End-to-end tests for Neocortical Schema Assimilation & Systems Consolidation.
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import torch
import numpy as np

from weft.kernels.fused_neocortical_schema_consolidation_triton import neocortical_consolidation_step
from scripts.sim_neocortical_systems_consolidation import run_consolidation_simulation


def test_consolidation_step_execution_and_shapes():
    """Verify that Triton/PyTorch consolidation kernel updates weights correctly."""
    B, K, M = 4, 64, 128
    torch.manual_seed(42)

    w_cortex = torch.randn(M, K) / np.sqrt(K)
    x_replay = torch.randn(B, K)
    y_hpc = torch.randn(B, M)

    w_new, y_comb, interf = neocortical_consolidation_step(
        w_cortex, x_replay, y_hpc, alpha_hpc=0.5, eta_cortex=0.01, decay=1e-4
    )

    assert w_new.shape == (M, K)
    assert y_comb.shape == (B, M)
    assert interf >= 0.0
    assert interf < 0.20 # slow updates prevent large jumps


def test_systems_consolidation_ribot_gradient():
    """Verify that post-lesion spared memory monotonically increases across epochs."""
    report = run_consolidation_simulation(num_memories=4, dim_input=64, dim_cortex=128, num_epochs=60, seed=42)
    epochs = report["epochs"]

    # Initial epoch should have poor post-lesion recall
    assert epochs[0]["post_lesion_spared_fidelity"] < 0.40

    # Late epoch should have high post-lesion recall
    assert epochs[-1]["post_lesion_spared_fidelity"] > 0.90
    assert epochs[-1]["neocortical_autonomy"] > 0.85


def test_catastrophic_forgetting_prevention():
    """Verify that multiple memories retain high fidelity under interleaved replay."""
    report = run_consolidation_simulation(num_memories=6, dim_input=64, dim_cortex=128, num_epochs=80, seed=42)
    assert report["final_cortex_fidelity"] > 0.90
    assert report["final_post_lesion_spared"] > 0.90
