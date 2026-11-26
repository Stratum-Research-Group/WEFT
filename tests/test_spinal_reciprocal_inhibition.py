"""
Unit & Integration Test Suite for Spinal Reciprocal Ia Inhibition Dynamics
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from weft.kernels.fused_spinal_reciprocal_inhibition_triton import fused_spinal_cpg_step
from scripts.sim_spinal_cpg_locomotion import run_spinal_cpg_sim

def test_spinal_cpg_simulation():
    res = run_spinal_cpg_sim()
    assert res["status"] == "success"
    assert res["reciprocal_inhibition_verified"] is True
    assert res["co_contraction_events"] == 0
    assert res["mean_suppressed_ext_hz"] < 2.0
    assert res["peak_flex_rate_hz"] > 80.0

def test_fused_spinal_cpg_step_reciprocal_exclusion():
    B = 2
    N_MN = 8

    # Case 1: High flexor drive, low extensor drive
    df = torch.full((B, N_MN), 120.0, dtype=torch.float32)
    de = torch.full((B, N_MN), 10.0, dtype=torch.float32)
    pf = torch.full((B, N_MN), 80.0, dtype=torch.float32)
    pe = torch.full((B, N_MN), 0.0, dtype=torch.float32)

    next_f, next_e = fused_spinal_cpg_step(df, de, pf, pe, g_Ia=1.2, g_renshaw=0.3)

    # Flexors active, extensors completely shut down by Ia reciprocal inhibition
    assert torch.all(next_f > 50.0)
    assert torch.all(next_e == 0.0)

    # Case 2: Inverted drives
    next_f_inv, next_e_inv = fused_spinal_cpg_step(de, df, pe, pf, g_Ia=1.2, g_renshaw=0.3)
    assert torch.all(next_e_inv > 50.0)
    assert torch.all(next_f_inv == 0.0)

def test_spinal_cpg_avx2_binary():
    bin_path = Path("build/weft-bench-spinal-cpg")
    assert bin_path.exists()
