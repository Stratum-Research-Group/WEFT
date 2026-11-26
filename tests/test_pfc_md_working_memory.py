"""
End-to-end tests for Prefrontal Working Memory Bistability & MD Thalamic Gating.
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import torch
import numpy as np

from weft.kernels.fused_pfc_md_thalamus_triton import pfc_md_working_memory_step
from scripts.sim_pfc_md_working_memory import run_pfc_wm_simulation


def test_pfc_md_step_execution_and_shapes():
    """Verify that Triton/PyTorch PFC-MD step kernel outputs valid shapes and positive rates."""
    B, N_PFC, N_MD = 2, 128, 64
    torch.manual_seed(42)

    x_pfc = torch.relu(torch.randn(B, N_PFC))
    x_md = torch.relu(torch.randn(B, N_MD))
    w_pfc = torch.randn(N_PFC, N_PFC) / np.sqrt(N_PFC)
    w_th_to_ctx = torch.randn(N_PFC, N_MD) / np.sqrt(N_MD)
    w_ctx_to_th = torch.randn(N_MD, N_PFC) / np.sqrt(N_PFC)

    x_pfc_next, x_md_next = pfc_md_working_memory_step(
        x_pfc, x_md, w_pfc, w_th_to_ctx, w_ctx_to_th, gamma_md=2.0, dt_over_tau=0.1, inh_bias=0.1
    )

    assert x_pfc_next.shape == (B, N_PFC)
    assert x_md_next.shape == (B, N_MD)
    assert (x_pfc_next >= 0.0).all()
    assert (x_md_next >= 0.0).all()


def test_pfc_md_delay_period_maintenance():
    """Verify that working memory items remain maintained across delay periods."""
    report = run_pfc_wm_simulation(num_pfc=256, num_md=64, delay_steps=40, seed=42)
    # Check that high gain cases maintain working memory
    for sim in report["simulations"]:
        if sim["gamma_md"] >= 1.2:
            assert sim["maintained"] is True
            assert sim["final_fidelity"] >= 0.90


def test_pfc_md_gain_amplification():
    """Verify that increasing MD thalamic gain strengthens prefrontal firing rate."""
    B, N_PFC, N_MD = 1, 64, 32
    x_pfc = torch.ones(B, N_PFC) * 0.5
    x_md = torch.ones(B, N_MD) * 0.5
    w_pfc = torch.ones(N_PFC, N_PFC) * 0.05
    w_th_to_ctx = torch.ones(N_PFC, N_MD) * 0.1
    w_ctx_to_th = torch.ones(N_MD, N_PFC) * 0.1

    # Low gain
    x_pfc_low, _ = pfc_md_working_memory_step(
        x_pfc, x_md, w_pfc, w_th_to_ctx, w_ctx_to_th, gamma_md=0.5, dt_over_tau=0.2, inh_bias=0.0
    )
    # High gain
    x_pfc_high, _ = pfc_md_working_memory_step(
        x_pfc, x_md, w_pfc, w_th_to_ctx, w_ctx_to_th, gamma_md=3.0, dt_over_tau=0.2, inh_bias=0.0
    )

    assert x_pfc_high.mean() > x_pfc_low.mean()
