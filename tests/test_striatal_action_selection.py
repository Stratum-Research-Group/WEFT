"""
End-to-end tests for Striatal D1/D2 Opponent Action Selection & Dopaminergic RPE.
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import torch
import numpy as np

from weft.kernels.fused_striatal_action_selection_triton import striatal_action_selection
from scripts.sim_striatal_action_selection import run_striatal_simulation


def test_striatal_action_selection_shapes_and_execution():
    """Verify that striatal D1/D2 selection kernel outputs valid shapes and tensors."""
    B, N_CTX, N_ACT = 2, 128, 16
    torch.manual_seed(42)

    x_ctx = torch.relu(torch.randn(B, N_CTX))
    w_d1 = torch.relu(torch.randn(N_ACT, N_CTX)) / np.sqrt(N_CTX)
    w_d2 = torch.relu(torch.randn(N_ACT, N_CTX)) / np.sqrt(N_CTX)
    delta_da = torch.tensor([0.8, -0.8]) # One positive RPE, one negative RPE

    y_d1, y_d2, action_gated = striatal_action_selection(
        x_ctx, w_d1, w_d2, delta_da, beta_d1=0.8, beta_d2=0.8, gpi_thresh=0.1
    )

    assert y_d1.shape == (B, N_ACT)
    assert y_d2.shape == (B, N_ACT)
    assert action_gated.shape == (B, N_ACT)
    assert (y_d1 >= 0.0).all()
    assert (y_d2 >= 0.0).all()


def test_dopamine_rpe_modulation_direction():
    """Verify that positive RPE boosts D1 drive and negative RPE boosts D2 drive."""
    B, N_CTX, N_ACT = 1, 64, 4
    x_ctx = torch.ones(B, N_CTX)
    w_d1 = torch.ones(N_ACT, N_CTX) * 0.1
    w_d2 = torch.ones(N_ACT, N_CTX) * 0.1

    # Positive RPE
    y_d1_pos, y_d2_pos, gated_pos = striatal_action_selection(
        x_ctx, w_d1, w_d2, torch.tensor([1.0]), beta_d1=0.8, beta_d2=0.8, gpi_thresh=0.05
    )
    # Negative RPE
    y_d1_neg, y_d2_neg, gated_neg = striatal_action_selection(
        x_ctx, w_d1, w_d2, torch.tensor([-1.0]), beta_d1=0.8, beta_d2=0.8, gpi_thresh=0.05
    )

    # Positive RPE should facilitate D1 over D2
    assert y_d1_pos.mean() > y_d2_pos.mean()
    assert gated_pos.sum() >= gated_neg.sum()

    # Negative RPE should facilitate D2 over D1
    assert y_d2_neg.mean() > y_d1_neg.mean()


def test_striatal_rl_policy_convergence():
    """Verify that reinforcement learning simulation converges to optimal choice."""
    report = run_striatal_simulation(num_ctx=64, num_actions=4, num_trials=100, seed=42)
    assert report["optimal_choice_ratio"] >= 0.80
    assert report["final_q_values"][1] > report["final_q_values"][0]
