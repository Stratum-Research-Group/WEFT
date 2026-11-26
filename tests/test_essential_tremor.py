"""
Regression Test Suite: Target #402
Essential Tremor (ET): Inferior Olivary Pacemaking & DRT Hyper-Oscillation
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group
"""

import pytest
import numpy as np
import torch

from weft.kernels.fused_essential_tremor_triton import compute_essential_tremor_triton


def test_essential_tremor_frequency_bounds():
    """Verify that inferior olivary pacemaker frequency remains strictly bounded between 4.0 and 8.0 Hz."""
    cav3_levels = torch.linspace(0.0, 1.0, 20)
    gap = torch.ones(20, dtype=torch.float32) * 0.85
    blk = torch.zeros(20, dtype=torch.float32)
    dbs = torch.zeros(20, dtype=torch.float32)

    freq, coh, pow_val, amp = compute_essential_tremor_triton(gap, cav3_levels, blk, dbs)

    assert torch.all(freq >= 4.0)
    assert torch.all(freq <= 8.0)


def test_gap_junction_coupling_generates_severe_tremor():
    """Verify that high gap-junction coupling and unblocked Cav3 produce high coherence and tremor amplitude > 2.5 cm."""
    gap = torch.tensor([0.85], dtype=torch.float32)
    cav = torch.tensor([0.80], dtype=torch.float32)
    blk = torch.tensor([0.0], dtype=torch.float32)
    dbs = torch.tensor([0.0], dtype=torch.float32)

    freq, coh, pow_val, amp = compute_essential_tremor_triton(gap, cav, blk, dbs)

    assert float(coh[0]) > 0.60  # Coherence > 60%
    assert float(amp[0]) > 2.5  # Tremor > 2.5 cm
    assert float(pow_val[0]) > 50.0  # Thalamic power > 50 uV^2


def test_cav3_blockade_rescues_tremor():
    """Verify that 90% Cav3 T-type calcium channel blockade decouples DRT coherence and reduces tremor < 1.0 cm."""
    gap = torch.tensor([0.85, 0.85], dtype=torch.float32)
    cav = torch.tensor([0.80, 0.80], dtype=torch.float32)
    blk = torch.tensor([0.0, 0.90], dtype=torch.float32)
    dbs = torch.tensor([0.0, 0.0], dtype=torch.float32)

    freq, coh, pow_val, amp = compute_essential_tremor_triton(gap, cav, blk, dbs)

    assert float(amp[0]) > 2.5
    assert float(amp[1]) < 1.0  # Rescued below 1.0 cm
    assert float(coh[1]) < float(coh[0])


def test_vim_dbs_desynchronization_arrests_tremor():
    """Verify that high-frequency VIM DBS desynchronizes DRT transmission and arrests tremor amplitude < 0.5 cm."""
    gap = torch.tensor([0.85, 0.85], dtype=torch.float32)
    cav = torch.tensor([0.80, 0.80], dtype=torch.float32)
    blk = torch.tensor([0.0, 0.0], dtype=torch.float32)
    dbs = torch.tensor([0.0, 1.0], dtype=torch.float32)

    freq, coh, pow_val, amp = compute_essential_tremor_triton(gap, cav, blk, dbs)

    assert float(amp[0]) > 2.5
    assert float(amp[1]) < 0.5  # Suppressed below 0.5 cm
    assert float(coh[1]) < 0.15  # Coherence reduced by DBS desync
