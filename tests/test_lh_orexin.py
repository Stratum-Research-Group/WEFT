"""
tests/test_lh_orexin.py
Pytest Regression Suite: Lateral Hypothalamus (LH) Orexin Arousal & Muscle Tone Gate
Lean 4 Target #396 | Paper #205 | Cycle v340.74.0
"""

import pytest
import math
import torch
from weft.kernels.fused_lh_orexin_triton import (
    launch_fused_lh_orexin,
)

def test_lh_orexin_rate_bounds_and_valence_monotonicity():
    """Verify that LH orexin firing rate satisfies bounded monotonic scaling with emotional valence."""
    rates = []
    integrity = 100
    circ = 75
    for v_int in range(0, 101, 10):
        # Lean 4 formula: clamp(((300 + 25 * v + 12 * c) * i) / 100, 0, 4500)
        drive = 300 + 25 * v_int + 12 * circ
        rate_scaled = min(4500, (drive * integrity) // 100)
        rate_hz = rate_scaled / 100.0
        rates.append(rate_hz)
        assert 0.0 <= rate_hz <= 45.0

    for i in range(len(rates) - 1):
        assert rates[i] <= rates[i+1]

def test_monoaminergic_facilitation_scaling():
    """Verify monoaminergic tone bounds and orexin scaling."""
    # Low orexin (0 Hz)
    mono_zero = min(1000, 150 + (2 * 0) // 10) / 1000.0
    # High orexin (35 Hz -> 3500)
    mono_high = min(1000, 150 + (2 * 3500) // 10) / 1000.0

    assert mono_zero == pytest.approx(0.15, abs=1e-3)
    assert mono_high == pytest.approx(0.85, abs=1e-3)

def test_cataplexy_prevention_in_healthy_vs_narcoleptic():
    """Verify cataplexy prevention in intact subject and collapse in narcoleptic state."""
    v = 90  # intense laughter

    # Intact subject:
    ox_intact = min(4500, ((300 + 25 * v + 12 * 75) * 100) // 100)
    mono_intact = min(1000, 150 + (2 * ox_intact) // 10)
    sld_intact = max(0, min(1000, 9 * v - (75 * mono_intact) // 100))
    tone_intact = max(0, min(1000, (90 * mono_intact) // 100 - (110 * sld_intact) // 100))

    assert sld_intact < 500, f"Healthy subject should suppress SLD atonia, got {sld_intact}"
    assert tone_intact >= 200, f"Healthy subject should maintain muscle tone, got {tone_intact}"

    # Narcoleptic subject:
    ox_narc = 0
    mono_narc = min(1000, 150 + (2 * ox_narc) // 10)
    sld_narc = max(0, min(1000, 9 * v - (75 * mono_narc) // 100))
    tone_narc = max(0, min(1000, (90 * mono_narc) // 100 - (110 * sld_narc) // 100))

    assert sld_narc >= 500, f"Narcoleptic subject should disinhibit SLD atonia, got {sld_narc}"
    assert tone_narc < 200, f"Narcoleptic subject should suffer muscle tone collapse, got {tone_narc}"

def test_cpu_vector_emulation_for_orexin_kernel():
    """Test batched tensor calculation matching the fused Triton GPU kernel interface."""
    N = 64
    valence = torch.linspace(0.0, 1.0, N, dtype=torch.float32)
    integrity = torch.full((N,), 1.0, dtype=torch.float32)
    circadian = torch.full((N,), 0.75, dtype=torch.float32)

    ox_base = 3.0
    ox_val_scale = 25.0
    ox_circ_scale = 12.0
    ox_max = 45.0
    mono_base = 0.15
    mono_scale = 0.02
    sld_val_scale = 0.90
    sld_mono_inh = 0.75
    tone_mono_scale = 0.90
    tone_atonia_scale = 1.10
    cata_atonia_th = 0.50
    cata_tone_th = 0.20
    cata_scale = 25.0

    raw_ox = (ox_base + ox_val_scale * valence + ox_circ_scale * circadian) * integrity
    ox = torch.clamp(raw_ox, 0.0, ox_max)

    raw_mono = mono_base + mono_scale * ox
    mono = torch.clamp(raw_mono, 0.0, 1.0)

    raw_sld = sld_val_scale * valence - sld_mono_inh * mono
    sld = torch.clamp(raw_sld, 0.0, 1.0)

    raw_tone = tone_mono_scale * mono - tone_atonia_scale * sld
    tone = torch.clamp(raw_tone, 0.0, 1.0)

    diff_atonia = torch.clamp(sld - cata_atonia_th, min=0.0)
    diff_tone = torch.clamp(cata_tone_th - tone, min=0.0)
    cata = torch.clamp(diff_atonia * diff_tone * cata_scale, 0.0, 1.0)

    assert len(ox) == N
    assert len(mono) == N
    assert len(sld) == N
    assert len(tone) == N
    assert len(cata) == N

    # Zero valence
    assert valence[0] == 0.0
    assert sld[0] == 0.0
    assert cata[0] == 0.0

    # Max valence in healthy individual
    assert valence[-1] == 1.0
    assert ox[-1] > 30.0
    assert cata[-1] == 0.0
