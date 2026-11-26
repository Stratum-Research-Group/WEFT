"""
Unit & Integration Test Suite for Cerebellar LTD & Motor Adaptation Dynamics
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from weft.kernels.fused_cerebellar_ltd_triton import fused_cerebellar_ltd_step, CerebellarPlasticityState
from scripts.sim_cerebellar_motor_adaptation import run_cerebellar_adaptation_sim

def test_cerebellar_adaptation_simulation():
    res = run_cerebellar_adaptation_sim()
    assert res["status"] == "success"
    assert res["adaptation_converged"] is True
    assert res["adaptation_ratio"] > 0.80
    assert res["final_error_deg"] < 0.50
    assert res["final_mean_weight"] < res["initial_mean_weight"]

def test_fused_cerebellar_ltd_step_plasticity():
    # Setup test weights [1 PC, 4 PF]
    weights = torch.tensor([[0.80, 0.80, 0.80, 0.80]], dtype=torch.float32)
    # Batch = 1: PF active on synapse 0 and 1, CF active
    pf_activity = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    cf_error = torch.tensor([[1.0]], dtype=torch.float32)

    new_weights, state = fused_cerebellar_ltd_step(
        weights, pf_activity, cf_error, eta_ltd=0.10, eta_ltp=0.01, w_min=0.1, w_max=1.0
    )

    # Synapses 0 and 1: Coincidence -> LTD
    assert new_weights[0, 0] < 0.80
    assert new_weights[0, 1] < 0.80
    assert state[0, 0] == CerebellarPlasticityState.LTD_DEPRESSING
    assert state[0, 1] == CerebellarPlasticityState.LTD_DEPRESSING

    # Synapses 2 and 3: Inactive -> Quiescent
    assert new_weights[0, 2] == 0.80
    assert new_weights[0, 3] == 0.80

def test_cerebellar_ltd_avx2_binary():
    bin_path = Path("build/weft-bench-cerebellar-ltd")
    assert bin_path.exists()
