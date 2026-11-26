"""
Unit & Integration Test Suite for Neuromuscular Junction Quantal Dynamics
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from weft.kernels.fused_nmj_vesicle_release_triton import fused_nmj_epp_step
from scripts.sim_nmj_twitch_dynamics import run_nmj_twitch_sim

def test_nmj_twitch_simulation():
    res = run_nmj_twitch_sim()
    assert res["status"] == "success"
    assert res["nmj_safety_factor_verified"] is True
    assert res["tetanic_fusion_ratio"] > 2.0
    assert res["transmission_reliability"] == 1.0
    assert res["peak_isolated_twitch_force"] > 0.2

def test_fused_nmj_epp_step_kernel():
    B = 2
    N_FIBERS = 4

    # High Ca2+ influx -> high release -> EPP exceeds threshold -> AP fired
    ca_high = torch.full((B, N_FIBERS), 0.95, dtype=torch.float32)
    pool = torch.full((B, N_FIBERS), 300.0, dtype=torch.float32)
    v_rest = torch.full((B, N_FIBERS), -85.0, dtype=torch.float32)

    quanta, epp, fired = fused_nmj_epp_step(ca_high, pool, v_rest, v_thresh_depol=25.0)

    assert torch.all(quanta > 50.0)
    assert torch.all(epp > 35.0)
    assert torch.all(fired == 1.0)

    # Low Ca2+ influx -> subthreshold release -> no AP fired
    ca_low = torch.full((B, N_FIBERS), 0.20, dtype=torch.float32)
    quanta_low, epp_low, fired_low = fused_nmj_epp_step(ca_low, pool, v_rest, v_thresh_depol=25.0)

    assert torch.all(epp_low < 15.0)
    assert torch.all(fired_low == 0.0)

def test_nmj_avx2_binary():
    bin_path = Path("build/weft-bench-nmj-quantal")
    assert bin_path.exists()
