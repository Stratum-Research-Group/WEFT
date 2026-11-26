"""
Unit & Integration Test Suite for Golgi Tendon Organ Dynamics & Autogenic Inhibition
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from weft.kernels.fused_gto_force_inhibition_triton import fused_gto_force_inhibition_step
from scripts.sim_gto_overload_protection import run_gto_overload_sim

def test_gto_overload_simulation():
    res = run_gto_overload_sim()
    assert res["status"] == "success"
    assert res["gto_protection_verified"] is True
    assert res["clasp_knife_shutoff_ratio"] == 1.0
    assert res["mean_post_overload_motor_drive_hz"] == 0.0

def test_fused_gto_kernel_clasp_knife_shutoff():
    B = 2
    N_T = 4

    # Case 1: Normal sub-critical tension
    t_normal = torch.full((B, N_T), 50.0, dtype=torch.float32)
    drive = torch.full((B, N_T), 90.0, dtype=torch.float32)

    f_Ib, inhib, net_alpha = fused_gto_force_inhibition_step(t_normal, drive, t_critical=120.0)

    assert torch.all(f_Ib > 20.0)
    assert torch.all(net_alpha > 0.0)
    assert torch.all(net_alpha < drive)  # Dampened by autogenic inhibition

    # Case 2: Overload tension >= critical
    t_over = torch.full((B, N_T), 135.0, dtype=torch.float32)
    f_Ib_over, inhib_over, net_alpha_over = fused_gto_force_inhibition_step(t_over, drive, t_critical=120.0)

    assert torch.all(net_alpha_over == 0.0)  # Clasp-knife shutoff

def test_gto_avx2_binary():
    bin_path = Path("build/weft-bench-gto-force")
    assert bin_path.exists()
