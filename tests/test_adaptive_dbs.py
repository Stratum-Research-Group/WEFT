"""
Regression Test Suite: Target #399
Adaptive Closed-Loop DBS (aDBS): Phase-Specific Desynchronization & Beta-Burst Truncation
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group
"""

import pytest
import numpy as np
import torch

from weft.kernels.fused_adbs_closed_loop_triton import compute_adbs_closed_loop_triton


def test_adbs_physiological_burst_preserved():
    """Verify that physiological beta bursts (<= 250 ms) do not trigger stimulation."""
    durations = torch.tensor([120.0, 180.0, 240.0, 250.0], dtype=torch.float32)
    phases = torch.tensor([45.0, 90.0, 180.0, 270.0], dtype=torch.float32)

    final_dur, stim_int, power = compute_adbs_closed_loop_triton(durations, phases)

    # Durations must remain unchanged and stimulation must be 0 mA
    assert torch.allclose(final_dur, durations)
    assert torch.all(stim_int == 0.0)
    assert torch.all(power < 52.0)


def test_adbs_pathological_burst_truncated():
    """Verify that pathological bursts (> 250 ms) are truncated below 350 ms."""
    durations = torch.tensor([300.0, 450.0, 600.0], dtype=torch.float32)
    phases = torch.tensor([180.0, 170.0, 200.0], dtype=torch.float32)  # Optimal phase window [150, 210]

    final_dur, stim_int, power = compute_adbs_closed_loop_triton(durations, phases)

    # Truncated to 275 ms under optimal phase
    assert torch.all(final_dur == 275.0)
    assert torch.all(stim_int == 2.5)
    assert torch.all(final_dur < 350.0)


def test_phase_targeted_differential_efficiency():
    """Verify that optimal phase (180 deg) produces shorter truncated burst and higher efficiency than off-phase (90 deg)."""
    durations = torch.tensor([400.0, 400.0], dtype=torch.float32)
    phases = torch.tensor([180.0, 90.0], dtype=torch.float32)

    final_dur, stim_int, power = compute_adbs_closed_loop_triton(durations, phases)

    assert float(final_dur[0]) == 275.0  # Optimal phase window
    assert float(final_dur[1]) == 325.0  # Off-phase window
    assert float(final_dur[0]) < float(final_dur[1])


def test_adbs_duty_cycle_and_energy_savings():
    """Verify that across a realistic Parkinsonian burst distribution, duty cycle remains < 45%."""
    t = torch.linspace(0.0, 10.0, 1000)
    # Burst envelope with physiological inter-burst intervals (pathological bursts occur ~38-40% of time)
    sin_vals = torch.sin(t * 1.5)
    raw_durations = torch.where(sin_vals > 0.50, 150.0 + sin_vals * 350.0, 100.0)
    phases = (t * 20.0 * 360.0) % 360.0

    final_dur, stim_int, power = compute_adbs_closed_loop_triton(raw_durations, phases)

    duty_cycle = float(torch.mean((stim_int > 0.0).float()))
    energy_savings = (1.0 - duty_cycle) * 100.0

    assert duty_cycle < 0.45, f"aDBS duty cycle ({duty_cycle:.2f}) must remain < 0.45"
    assert energy_savings > 55.0, f"Energy savings ({energy_savings:.1f}%) must exceed 55%"
