"""
End-to-end tests for Subicular Gating, SWR Burst Dynamics, and Neocortical Replay.
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import torch
import numpy as np

from weft.kernels.fused_subiculum_burst_ripple_triton import subiculum_replay_broadcast
from scripts.sim_subiculum_swr_replay import run_subiculum_simulation


def test_subiculum_burst_generation_and_routing_shapes():
    """Verify that Subiculum SWR burst generation and multi-target routing executes properly."""
    B, N_SUB, N_MPFC, N_EC = 2, 128, 64, 64
    torch.manual_seed(42)

    v_ca1 = torch.randn(B, N_SUB) * 10.0 - 50.0 # Some above -55 mV, some below
    w_mpfc = torch.randn(N_MPFC, N_SUB) / np.sqrt(N_SUB)
    w_ec = torch.randn(N_EC, N_SUB) / np.sqrt(N_SUB)

    y_mpfc, y_ec, burst_act = subiculum_replay_broadcast(
        v_ca1, w_mpfc, w_ec, swr_active=True, ripple_freq_hz=200.0, time_ms=25.0
    )

    assert y_mpfc.shape == (B, N_MPFC)
    assert y_ec.shape == (B, N_EC)
    assert burst_act.shape == (B, N_SUB)
    assert (burst_act >= 0.0).all()
    assert (y_mpfc >= 0.0).all()
    assert (y_ec >= 0.0).all()


def test_subiculum_burst_threshold_gating():
    """Verify that hyperpolarized neurons (< -55 mV) do not trigger burst firing."""
    B, N_SUB = 1, 64
    v_hyper = torch.full((B, N_SUB), -65.0) # Well below -55 mV threshold
    w_mpfc = torch.ones(32, N_SUB)
    w_ec = torch.ones(32, N_SUB)

    _, _, burst_act = subiculum_replay_broadcast(
        v_hyper, w_mpfc, w_ec, swr_active=True, ripple_freq_hz=200.0, time_ms=10.0
    )

    assert torch.allclose(burst_act, torch.zeros_like(burst_act), atol=1e-5)


def test_subiculum_swr_compression_and_replay_fidelity():
    """Verify that SWR replay yields 100% trajectory fidelity across forward and reverse runs."""
    report = run_subiculum_simulation(num_sub=256, num_mpfc=128, num_ec_deep=128, trajectory_length=12, seed=42)
    assert report["mean_forward_replay_fidelity"] >= 0.99
    assert report["mean_reverse_replay_fidelity"] >= 0.99
    assert report["compression_factor"] == 15.0


def test_subiculum_quiescent_attenuation():
    """Verify that inactive SWR dampens broadcasting drive significantly."""
    B, N_SUB = 1, 64
    v_depol = torch.full((B, N_SUB), -45.0) # Depolarized
    w_mpfc = torch.ones(32, N_SUB)
    w_ec = torch.ones(32, N_SUB)

    _, _, burst_active = subiculum_replay_broadcast(v_depol, w_mpfc, w_ec, swr_active=True, time_ms=0.0)
    _, _, burst_inactive = subiculum_replay_broadcast(v_depol, w_mpfc, w_ec, swr_active=False, time_ms=0.0)

    # Active SWR should produce significantly higher drive than inactive
    assert burst_active.mean() > burst_inactive.mean() * 3.0
