"""
End-to-end tests for Pedunculopontine Nucleus (PPN) Locomotor Gating & CPG Stepping.
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import torch
import numpy as np

from weft.kernels.fused_ppn_locomotor_gating_triton import ppn_locomotor_gating
from scripts.sim_ppn_locomotor_gating import run_ppn_simulation


def test_ppn_locomotor_gating_shapes_and_execution():
    """Verify that PPN locomotor gating kernel executes properly and outputs valid shapes."""
    B, N_PPN, N_GPI = 2, 64, 32
    torch.manual_seed(42)

    x_gpi = torch.relu(torch.randn(B, N_GPI)) * 0.5
    w_gpi = torch.relu(torch.randn(N_PPN, N_GPI)) / np.sqrt(N_GPI)
    i_drive = torch.ones(B, N_PPN) * 2.0

    y_glu, y_ach, step_freq = ppn_locomotor_gating(
        x_gpi, w_gpi, i_drive, theta_loco=0.2, alpha_step=0.08, ach_gain=0.85
    )

    assert y_glu.shape == (B, N_PPN)
    assert y_ach.shape == (B, N_PPN)
    assert step_freq.shape == (B,)
    assert (y_glu >= 0.0).all()
    assert (y_ach >= 0.0).all()
    assert (step_freq >= 0.0).all()


def test_ppn_disinhibition_versus_inhibition():
    """Verify that full disinhibition triggers high stepping frequency while heavy inhibition causes arrest."""
    B, N_PPN, N_GPI = 1, 64, 32
    w_gpi = torch.ones(N_PPN, N_GPI) * 0.1
    i_drive = torch.ones(B, N_PPN) * 2.0

    # Disinhibited (GPi = 0)
    x_disinh = torch.zeros(B, N_GPI)
    _, _, freq_high = ppn_locomotor_gating(x_disinh, w_gpi, i_drive, theta_loco=0.2, alpha_step=1.0)

    # Heavily inhibited (GPi = 5.0)
    x_inhib = torch.ones(B, N_GPI) * 5.0
    _, _, freq_zero = ppn_locomotor_gating(x_inhib, w_gpi, i_drive, theta_loco=0.2, alpha_step=1.0)

    assert freq_high.item() > 1.0
    assert freq_zero.item() == 0.0


def test_ppn_locomotor_kinematics_spectrum():
    """Verify that simulated gait spectrum traverses from gallop to complete freezing."""
    report = run_ppn_simulation(num_ppn=64, num_gpi=32, num_levels=8, seed=42)
    sims = report["simulations"]

    # First level (0 brake) should be gallop
    assert sims[0]["gait"] == "GALLOP"
    assert sims[0]["step_frequency_hz"] > 3.5

    # Last level (high brake) should be arrest freezing
    assert sims[-1]["gait"] == "ARREST_FREEZING"
    assert sims[-1]["step_frequency_hz"] == 0.0
