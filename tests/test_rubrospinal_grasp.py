"""
Unit & Integration Test Suite for Magnocellular Red Nucleus & Rubrospinal Tract Dynamics
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from weft.kernels.fused_rubrospinal_velocity_triton import fused_rubrospinal_velocity_step
from scripts.sim_rubrospinal_grasp_kinematics import run_rubrospinal_grasp_sim

def test_rubrospinal_grasp_simulation():
    res = run_rubrospinal_grasp_sim()
    assert res["status"] == "success"
    assert res["precision_grasp_verified"] is True
    assert res["peak_velocity_m_s"] > 0.70
    assert res["peak_rnm_discharge_hz"] > 150.0
    assert res["terminal_tremor_amplitude_mm" if "terminal_tremor_amplitude_mm" in res else "tremor_amplitude_mm"] < 3.0
    assert res["final_position_error_cm"] < 2.0

def test_fused_rubrospinal_kernel():
    B = 2
    N_U = 4

    # High velocity reach condition
    v_fast = torch.full((B, N_U), 0.90, dtype=torch.float32)
    x_inter = torch.full((B, N_U), 100.0, dtype=torch.float32)
    x_m1 = torch.full((B, N_U), 80.0, dtype=torch.float32)

    rnm, flex = fused_rubrospinal_velocity_step(v_fast, x_inter, x_m1)

    assert torch.all(rnm > 120.0)
    assert torch.all(flex > 100.0)

    # Stationary rest
    v_zero = torch.zeros((B, N_U), dtype=torch.float32)
    x_inter_0 = torch.zeros((B, N_U), dtype=torch.float32)
    x_m1_0 = torch.zeros((B, N_U), dtype=torch.float32)

    rnm_0, flex_0 = fused_rubrospinal_velocity_step(v_zero, x_inter_0, x_m1_0)
    assert torch.allclose(rnm_0, torch.full_like(rnm_0, 35.0), atol=1e-3)
    assert torch.allclose(flex_0, torch.full_like(flex_0, 20.0), atol=1e-3)

def test_rubrospinal_avx2_binary():
    bin_path = Path("build/weft-bench-rubrospinal")
    assert bin_path.exists()
