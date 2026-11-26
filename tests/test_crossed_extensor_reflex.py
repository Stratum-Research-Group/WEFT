"""
Unit & Integration Test Suite for Crossed-Extensor Reflex Dynamics
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from weft.kernels.fused_crossed_extensor_reflex_triton import fused_crossed_extensor_step
from scripts.sim_crossed_extensor_withdrawal import run_crossed_extensor_sim

def test_crossed_extensor_simulation():
    res = run_crossed_extensor_sim()
    assert res["status"] == "success"
    assert res["postural_preservation_verified"] is True
    assert res["mean_ipsi_flex_hz"] > 120.0
    assert res["mean_ipsi_ext_hz"] == 0.0
    assert res["mean_contra_ext_hz"] > 160.0
    assert res["mean_contra_flex_hz"] == 0.0

def test_fused_crossed_extensor_kernel():
    B = 2
    N_L = 4

    # High nociceptive pain input
    e_nox = torch.full((B, N_L), 100.0, dtype=torch.float32)
    d_if = torch.full((B, N_L), 15.0, dtype=torch.float32)
    d_ie = torch.full((B, N_L), 55.0, dtype=torch.float32)
    d_cf = torch.full((B, N_L), 15.0, dtype=torch.float32)
    d_ce = torch.full((B, N_L), 55.0, dtype=torch.float32)

    mn_if, mn_ie, mn_ce, mn_cf = fused_crossed_extensor_step(
        e_nox, d_if, d_ie, d_cf, d_ce
    )

    # Verify withdrawal: Ipsi flex high, ipsi ext 0
    assert torch.all(mn_if > 100.0)
    assert torch.all(mn_ie == 0.0)

    # Verify contralateral support: Contra ext high, contra flex 0
    assert torch.all(mn_ce > 150.0)
    assert torch.all(mn_cf == 0.0)

def test_crossed_extensor_avx2_binary():
    bin_path = Path("build/weft-bench-crossed-extensor")
    assert bin_path.exists()
