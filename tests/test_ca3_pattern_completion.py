"""
End-to-end tests for Hippocampal CA3 Auto-Associative Pattern Completion & Attractor Dynamics.
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import torch
import numpy as np

from weft.kernels.fused_ca3_autoassociative_triton import ca3_recurrent_settle
from scripts.sim_ca3_pattern_completion import run_ca3_simulation


def test_ca3_recurrent_settling_convergence():
    """Verify that PyTorch/Triton recurrent settling converges within max steps."""
    N = 256
    torch.manual_seed(42)
    # Generate Hebbian weight matrix
    pattern = torch.sign(torch.randn(1, N))
    w_rc = torch.matmul(pattern.t(), pattern) / N
    w_rc.fill_diagonal_(0.0)

    # Corrupt pattern (30% masked)
    cue = pattern.clone()
    cue[:, : int(0.3 * N)] = 0.0

    x_settled, steps, energy = ca3_recurrent_settle(cue, w_rc, max_steps=20, beta=0.25)
    assert steps > 0
    assert steps <= 20
    # Settled pattern should be close to ground truth pattern
    overlap = torch.mean(torch.sign(x_settled) * pattern).item()
    assert overlap > 0.90, f"Expected overlap > 0.90, got {overlap}"


def test_ca3_pattern_completion_fidelity():
    """Verify that simulated cue occlusion up to 60% yields perfect attractor recovery."""
    report = run_ca3_simulation(num_neurons=256, num_patterns=4, seed=42)
    for sim in report["simulations"]:
        if sim["occlusion_ratio"] <= 0.60:
            assert sim["final_hamming_distance"] == 0, (
                f"Failed pattern completion at occlusion {sim['occlusion_ratio']}: "
                f"hamming distance = {sim['final_hamming_distance']}"
            )
            assert sim["retrieval_fidelity"] == 1.0


def test_ca3_monotonic_lyapunov_descent():
    """Verify that recurrent settling satisfies monotonic Lyapunov energy descent."""
    report = run_ca3_simulation(num_neurons=256, num_patterns=4, seed=42)
    for sim in report["simulations"]:
        assert sim["monotonic_energy_descent"] is True, (
            f"Monotonic energy descent violated at occlusion {sim['occlusion_ratio']}"
        )


def test_ca3_energy_lower_bound():
    """Verify that final settled energy is strictly lower than or equal to initial energy."""
    report = run_ca3_simulation(num_neurons=256, num_patterns=4, seed=42)
    for sim in report["simulations"]:
        assert sim["energy_final"] <= sim["energy_start"] + 1e-4
