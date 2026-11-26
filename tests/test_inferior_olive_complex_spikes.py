"""
End-to-end tests for Inferior Olive Gap-Junction Coupling & Climbing Fiber Plasticity.
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import torch
import numpy as np

from weft.kernels.fused_inferior_olive_climbing_fiber_triton import inferior_olive_climbing_fiber_step
from scripts.sim_inferior_olive_complex_spikes import run_inferior_olive_simulation


def test_inferior_olive_step_shapes_and_execution():
    """Verify that Inferior Olive gap junction kernel executes properly and outputs valid shapes."""
    B, N = 2, 64
    torch.manual_seed(42)

    v_curr = torch.randn(B, N) * 5.0 - 60.0
    i_error = torch.relu(torch.randn(B, N)) * 10.0
    # Symmetric Laplacian: L = D - A
    A = torch.eye(N)
    L = torch.eye(N) - A

    v_next, spike_mask, coherence = inferior_olive_climbing_fiber_step(
        v_curr, i_error, L, g_gap=0.5, v_thresh=-45.0, dt_over_tau=0.1
    )

    assert v_next.shape == (B, N)
    assert spike_mask.shape == (B, N)
    assert 0.0 <= coherence <= 1.0


def test_inferior_olive_variance_contraction():
    """Verify that gap-junction coupling reduces membrane potential variance across time."""
    B, N = 1, 64
    v_hetero = torch.linspace(-70.0, -50.0, N).unsqueeze(0)
    i_zero = torch.zeros(B, N)

    # Ring Laplacian
    A = torch.zeros(N, N)
    for i in range(N):
        A[i, (i + 1) % N] = 1.0
        A[i, (i - 1) % N] = 1.0
    L = 2.0 * torch.eye(N) - A

    v_step = v_hetero.clone()
    var_init = torch.var(v_step).item()

    for _ in range(10):
        v_step, _, _ = inferior_olive_climbing_fiber_step(v_step, i_zero, L, g_gap=0.8, dt_over_tau=0.1)

    var_final = torch.var(v_step).item()
    assert var_final < var_init


def test_inferior_olive_simulation_coherence():
    """Verify that full olivary simulation achieves high phase synchronization order."""
    report = run_inferior_olive_simulation(num_neurons=64, sim_steps=60, seed=42)
    # Strong coupling should exhibit synchrony > 90%
    strong_sim = report["simulations"][-1]
    assert strong_sim["synchrony_order_parameter"] > 0.90
    assert strong_sim["final_dispersion_mv"] < strong_sim["initial_dispersion_mv"]
