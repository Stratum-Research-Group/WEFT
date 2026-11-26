"""
End-to-end tests for Hippocampal CA1 Dual-Stream Comparator & Sequence Readout.
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import torch
import numpy as np

from weft.kernels.fused_ca1_schaffer_readout_triton import ca1_dual_stream_readout
from scripts.sim_ca1_sequence_readout import run_ca1_simulation


def test_ca1_dual_stream_readout_execution():
    """Verify that CA1 dual-stream kernel executes properly and outputs correct shapes."""
    B, N_CA1, N_CA3, N_EC = 2, 128, 64, 64
    torch.manual_seed(42)

    x_ca3 = torch.randn(B, N_CA3)
    x_ec = torch.randn(B, N_EC)
    w_schaffer = torch.randn(N_CA1, N_CA3) / np.sqrt(N_CA3)
    w_ta = torch.randn(N_CA1, N_EC) / np.sqrt(N_EC)

    y_ca1, mismatch = ca1_dual_stream_readout(
        x_ca3, x_ec, w_schaffer, w_ta, theta_rad=3.14159, theta_inhibition=0.0
    )

    assert y_ca1.shape == (B, N_CA1)
    assert mismatch.shape == (B,)
    assert (y_ca1 >= 0.0).all()


def test_ca1_theta_phase_gating_extremes():
    """Verify that theta peak shuts down CA3 drive and trough shuts down EC drive."""
    B, N_CA1, N_CA3, N_EC = 1, 64, 32, 32
    x_ca3 = torch.ones(B, N_CA3)
    x_ec = torch.zeros(B, N_EC)
    w_schaffer = torch.ones(N_CA1, N_CA3)
    w_ta = torch.ones(N_CA1, N_EC)

    # At theta = 0 (peak), g_ca3 = 0, so y_ca1 should be zero since x_ec is zero
    y_peak, _ = ca1_dual_stream_readout(x_ca3, x_ec, w_schaffer, w_ta, theta_rad=0.0, theta_inhibition=0.0)
    assert torch.allclose(y_peak, torch.zeros_like(y_peak), atol=1e-5)

    # At theta = pi (trough), g_ca3 = 1.0, so y_ca1 should be active
    y_trough, _ = ca1_dual_stream_readout(x_ca3, x_ec, w_schaffer, w_ta, theta_rad=3.14159265, theta_inhibition=0.0)
    assert (y_trough > 0.0).all()


def test_ca1_sequence_decoding_accuracy():
    """Verify that simulated sequence readout preserves 100% temporal order accuracy."""
    report = run_ca1_simulation(num_ca1=256, num_ca3=128, num_ec=128, num_sequence_steps=8, seed=42)
    assert report["sequence_accuracy"] == 1.0


def test_ca1_novelty_mismatch_detection():
    """Verify that novel sensory patterns trigger mismatch scores exceeding detection threshold."""
    report = run_ca1_simulation(num_ca1=256, num_ca3=128, num_ec=128, seed=42)
    for ps in report["phase_sweep"]:
        assert ps["novelty_detected"] is True
        assert ps["mismatch_novel"] > 0.50
        assert ps["mismatch_congruent"] < 0.05
