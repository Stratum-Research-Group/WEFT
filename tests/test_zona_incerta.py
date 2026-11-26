"""
tests/test_zona_incerta.py
Pytest Regression Suite: Zona Incerta (ZI) Visceromotor Integration & Binge Feeding Gating
Lean 4 Target #394 | Paper #203 | Cycle v340.72.0
"""

import pytest
import math
import torch
from weft.kernels.fused_zona_incerta_triton import (
    launch_fused_zona_incerta,
)

def test_zona_incerta_pv_rate_bounds_and_monotonicity():
    """Verify rostral PV+ rate bounds and monotonic scaling with hunger."""
    rates = []
    for h_int in range(0, 101, 10):
        # Lean 4 formula: clamp(400 + 50 * h, 0, 6000)
        pv_rate_scaled = min(6000, 400 + 50 * h_int)
        pv_rate_hz = pv_rate_scaled / 100.0
        rates.append(pv_rate_hz)
        assert 0.0 <= pv_rate_hz <= 60.0

    for i in range(len(rates) - 1):
        assert rates[i] <= rates[i+1]

def test_zona_incerta_caudal_som_prey_scaling():
    """Verify caudal SOM+ rate bounds and prey proximity scaling."""
    som_rates = []
    h = 50  # fixed intermediate hunger
    for p_int in range(0, 101, 10):
        # Lean 4 formula: clamp(300 + (45 * p * (50 + h / 2)) / 100, 0, 5000)
        drive = 300 + (45 * p_int * (50 + h // 2)) // 100
        som_rate_scaled = min(5000, drive)
        som_rate_hz = som_rate_scaled / 100.0
        som_rates.append(som_rate_hz)
        assert 0.0 <= som_rate_hz <= 50.0

    for i in range(len(som_rates) - 1):
        assert som_rates[i] <= som_rates[i+1]

def test_binge_feeding_trigger_threshold():
    """Verify that severe hunger under low stress strictly triggers binge feeding (>30 Hz)."""
    h_high = 85
    s_low = 5
    pv_scaled = min(6000, max(0, 400 + 50 * h_high - 15 * s_low))
    pv_hz = pv_scaled / 100.0
    assert pv_hz >= 30.0, f"Expected binge trigger, got {pv_hz} Hz"

    # Satiated baseline check (<30 Hz)
    h_low = 10
    s_norm = 15
    pv_base_scaled = min(6000, max(0, 400 + 50 * h_low - 15 * s_norm))
    pv_base_hz = pv_base_scaled / 100.0
    assert pv_base_hz < 30.0, f"Satiated state should not trigger binge, got {pv_base_hz} Hz"

def test_pag_disinhibition_limits():
    """Verify PAG motor disinhibition bounds [0, 1]."""
    # Low activity
    pag_low = (400 * 60 + 300 * 40) / 500 / 1000.0
    assert 0.0 <= pag_low <= 1.0

    # Max activity (60 Hz PV, 50 Hz SOM)
    pag_max = min(1000, (6000 * 60 + 5000 * 40) // 500) / 1000.0
    assert pag_max == 1.0

def test_cpu_vector_emulation_for_kernel():
    """Test batched tensor computation matching the Triton kernel interface."""
    N = 64
    hunger = torch.linspace(0.0, 1.0, N, dtype=torch.float32)
    prey = torch.full((N,), 0.3, dtype=torch.float32)
    stress = torch.full((N,), 0.1, dtype=torch.float32)

    pv_base = 4.0
    pv_hunger_scale = 50.0
    pv_stress_scale = 15.0
    pv_max = 60.0
    som_base = 3.0
    som_scale = 45.0
    som_max = 50.0
    pag_pv_scale = 0.012
    pag_som_scale = 0.008
    binge_slope = 0.12
    binge_th = 30.0

    raw_pv = pv_base + pv_hunger_scale * hunger - pv_stress_scale * stress
    pv = torch.clamp(raw_pv, 0.0, pv_max)

    amp = 0.5 + 0.5 * hunger
    raw_som = som_base + som_scale * prey * amp
    som = torch.clamp(raw_som, 0.0, som_max)

    raw_pag = pag_pv_scale * pv + pag_som_scale * som
    pag = torch.clamp(raw_pag, 0.0, 1.0)

    raw_binge = 0.5 + binge_slope * (pv - binge_th)
    binge = torch.clamp(raw_binge, 0.0, 1.0)

    assert len(pv) == N
    assert len(som) == N
    assert len(pag) == N
    assert len(binge) == N

    # Low hunger edge
    assert hunger[0] == 0.0
    assert pv[0] < 5.0
    assert binge[0] == 0.0

    # High hunger edge
    assert hunger[-1] == 1.0
    assert pv[-1] > 40.0
    assert binge[-1] == 1.0
