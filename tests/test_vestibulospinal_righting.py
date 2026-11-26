"""
Unit & Integration Test Suite for Vestibulospinal Postural Tone & Righting Reflex Dynamics
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from weft.kernels.fused_vestibulospinal_posture_triton import fused_vestibulospinal_step
from scripts.sim_vestibulospinal_righting import run_vestibulospinal_sim

def test_vestibulospinal_simulation():
    res = run_vestibulospinal_sim()
    assert res["status"] == "success"
    assert res["righting_reflex_verified"] is True
    assert res["peak_lvst_drive_hz"] > 150.0
    assert abs(res["final_tilt_deg"]) < 2.0

def test_fused_vestibulospinal_kernel():
    B = 2
    N_U = 4

    # Case 1: Substantial lateral head tilt (0.4 rad ~ 23 deg) and angular velocity
    tilt_rad = torch.full((B, N_U), 0.40, dtype=torch.float32)
    ang_vel = torch.full((B, N_U), 60.0, dtype=torch.float32)

    f_oto, f_can, lvst, mvst = fused_vestibulospinal_step(tilt_rad, ang_vel)

    assert torch.all(f_oto > 90.0)
    assert torch.all(f_can > 120.0)
    assert torch.all(lvst > 100.0)  # Vigorous antigravity extensor drive
    assert torch.all(mvst > 40.0)   # Active vestibulocollic neck stabilization

    # Case 2: Upright vertical stationary position
    tilt_zero = torch.zeros((B, N_U), dtype=torch.float32)
    vel_zero = torch.zeros((B, N_U), dtype=torch.float32)

    f_oto_0, f_can_0, lvst_0, mvst_0 = fused_vestibulospinal_step(tilt_zero, vel_zero)
    assert torch.allclose(f_oto_0, torch.full_like(f_oto_0, 60.0), atol=1e-3)
    assert torch.allclose(lvst_0, torch.full_like(lvst_0, 45.0), atol=1e-3)

def test_vestibulospinal_avx2_binary():
    bin_path = Path("build/weft-bench-vestibulospinal")
    assert bin_path.exists()
