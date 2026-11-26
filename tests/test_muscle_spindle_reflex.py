"""
Unit & Integration Test Suite for Muscle Spindle & Monosynaptic Reflex Dynamics
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from weft.kernels.fused_muscle_spindle_afferent_triton import fused_muscle_spindle_reflex_step
from scripts.sim_myotatic_stretch_reflex import run_stretch_reflex_sim

def test_stretch_reflex_simulation():
    res = run_stretch_reflex_sim()
    assert res["status"] == "success"
    assert res["stretch_reflex_verified"] is True
    assert res["peak_ia_discharge_hz"] > 140.0
    assert res["peak_reflex_motor_hz"] > 140.0
    assert abs(res["final_restored_length"] - 1.0) < 0.05

def test_fused_muscle_spindle_kernel():
    B = 2
    N_S = 4

    # High velocity lengthening perturbation
    L_stretched = torch.full((B, N_S), 1.15, dtype=torch.float32)
    v_fast = torch.full((B, N_S), 3.0, dtype=torch.float32)
    g_dyn = torch.full((B, N_S), 0.70, dtype=torch.float32)
    g_stat = torch.full((B, N_S), 0.50, dtype=torch.float32)

    f_Ia, f_II, alpha_mn = fused_muscle_spindle_reflex_step(L_stretched, v_fast, g_dyn, g_stat)

    # Robust Ia discharge and reflex recruitment
    assert torch.all(f_Ia > 150.0)
    assert torch.all(alpha_mn > 150.0)

    # Rest / isometric state: v = 0, L = L0
    L_rest = torch.full((B, N_S), 1.0, dtype=torch.float32)
    v_zero = torch.zeros((B, N_S), dtype=torch.float32)

    f_Ia_rest, f_II_rest, alpha_mn_rest = fused_muscle_spindle_reflex_step(L_rest, v_zero, g_dyn, g_stat)
    assert torch.all(f_Ia_rest < 40.0)
    assert torch.all(alpha_mn_rest == 0.0)

def test_spindle_avx2_binary():
    bin_path = Path("build/weft-bench-muscle-spindle")
    assert bin_path.exists()
