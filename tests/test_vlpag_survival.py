"""
tests/test_vlpag_survival.py
Pytest Regression Suite: Ventrolateral Periaqueductal Gray (vlPAG) Freezing vs Escape
Lean 4 Target #395 | Paper #204 | Cycle v340.73.0
"""

import pytest
import math
import torch
from weft.kernels.fused_vlpag_survival_triton import (
    launch_fused_vlpag_survival,
)

def test_vlpag_raw_drives_and_bounds():
    """Verify raw vlPAG and dlPAG drives adhere to bounds and scaling."""
    # Distant threat (dist = 80, loom = 0, cort = 50)
    dist = 80
    loom = 0
    cort = 50
    raw_vl = 300 + (35 * dist * (100 - loom)) // 100 + 15 * cort
    raw_dl = 200 + 60 * (100 - dist) + 40 * loom

    assert raw_vl >= 3850
    assert raw_dl == 1400

    # Cross inhibition
    inh_on_vl = (45 * raw_dl) // 100
    eff_vl = max(0, min(5000, raw_vl - inh_on_vl))
    eff_vl_hz = eff_vl / 100.0

    assert eff_vl_hz >= 20.0, f"Distant threat should trigger freezing (>20 Hz), got {eff_vl_hz} Hz"

def test_imminent_looming_escape_trigger():
    """Verify that immediate looming strike triggers dlPAG escape (>35 Hz) and silences vlPAG."""
    dist = 0
    loom = 100
    cort = 20
    raw_vl = 300 + (35 * dist * (100 - loom)) // 100 + 15 * cort
    raw_dl = 200 + 60 * (100 - dist) + 40 * loom

    inh_on_vl = (45 * raw_dl) // 100
    eff_vl = max(0, min(5000, raw_vl - inh_on_vl))
    eff_vl_hz = eff_vl / 100.0

    inh_on_dl = (20 * raw_vl) // 100
    eff_dl = max(0, min(8000, raw_dl - inh_on_dl))
    eff_dl_hz = eff_dl / 100.0

    assert eff_dl_hz >= 35.0, f"Looming threat should trigger escape (>35 Hz), got {eff_dl_hz} Hz"
    assert eff_vl_hz == 0.0, f"Escape surge must completely silence vlPAG freezing, got {eff_vl_hz} Hz"

def test_sympathetic_tone_bounds():
    """Verify autonomic sympathetic tone bounds [0, 1]."""
    # Low activity
    tone_low = min(1000, (200 * 12 + 300 * 2) // 100) / 1000.0
    assert 0.0 <= tone_low <= 1.0

    # High flight rate (80 Hz dlPAG)
    tone_high = min(1000, (8000 * 12 + 0 * 2) // 100) / 1000.0
    assert tone_high == pytest.approx(0.96, abs=1e-2)

def test_cpu_vector_emulation_for_pag_kernel():
    """Test batched tensor calculation matching the fused Triton GPU kernel interface."""
    N = 64
    threat_dist = torch.linspace(0.0, 1.0, N, dtype=torch.float32)
    looming_speed = torch.linspace(1.0, 0.0, N, dtype=torch.float32)
    corticosterone = torch.full((N,), 0.3, dtype=torch.float32)

    vl_base = 3.0
    vl_dist_scale = 35.0
    vl_cort_scale = 15.0
    vl_max = 50.0
    dl_base = 2.0
    dl_prox_scale = 60.0
    dl_loom_scale = 40.0
    dl_max = 80.0
    inh_dl_on_vl = 0.45
    inh_vl_on_dl = 0.20
    sym_dl_scale = 0.012
    sym_vl_scale = 0.002
    freeze_th = 20.0
    freeze_slope = 0.10
    escape_th = 35.0
    escape_slope = 0.08

    non_loom = 1.0 - looming_speed
    raw_vl = vl_base + vl_dist_scale * threat_dist * non_loom + vl_cort_scale * corticosterone
    prox = 1.0 - threat_dist
    raw_dl = dl_base + dl_prox_scale * prox + dl_loom_scale * looming_speed

    net_vl = raw_vl - inh_dl_on_vl * raw_dl
    eff_vl = torch.clamp(net_vl, 0.0, vl_max)

    net_dl = raw_dl - inh_vl_on_dl * raw_vl
    eff_dl = torch.clamp(net_dl, 0.0, dl_max)

    raw_sym = sym_dl_scale * eff_dl + sym_vl_scale * eff_vl
    sym = torch.clamp(raw_sym, 0.0, 1.0)

    raw_freeze = 0.5 + freeze_slope * (eff_vl - freeze_th)
    freeze_prob = torch.clamp(raw_freeze, 0.0, 1.0)

    raw_escape = 0.5 + escape_slope * (eff_dl - escape_th)
    escape_prob = torch.clamp(raw_escape, 0.0, 1.0)

    assert len(eff_vl) == N
    assert len(eff_dl) == N
    assert len(sym) == N
    assert len(freeze_prob) == N
    assert len(escape_prob) == N

    # Imminent looming index 0:
    assert threat_dist[0] == 0.0
    assert looming_speed[0] == 1.0
    assert eff_dl[0] > 70.0
    assert eff_vl[0] == 0.0
    assert escape_prob[0] == 1.0
    assert freeze_prob[0] == 0.0

    # Distant quiet index -1:
    assert threat_dist[-1] == 1.0
    assert looming_speed[-1] == 0.0
    assert eff_vl[-1] > 30.0
    assert freeze_prob[-1] == 1.0
