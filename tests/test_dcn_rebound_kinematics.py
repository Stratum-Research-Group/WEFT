"""
Unit & Integration Test Suite for DCN Post-Inhibitory Rebound Dynamics
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from weft.kernels.fused_dcn_rebound_burst_triton import fused_dcn_rebound_burst, DCNFiringMode
from scripts.sim_dcn_rebound_kinematics import run_dcn_rebound_sim

def test_dcn_simulation_kinematics():
    results = run_dcn_rebound_sim()
    assert results["status"] == "success"
    assert results["rebound_invariant_verified"] is True
    assert results["rebound_burst_peak_hz"] >= 200.0
    assert results["burst_acceleration_ratio"] >= 10.0
    assert results["baseline_rate_hz"] <= 20.0

def test_dcn_rebound_burst_logic_pure_python():
    # Purkinje paused (0 Hz) and primed (0.85) -> Explosive Rebound
    pc_rate = torch.tensor([0.0, 80.0, 20.0], dtype=torch.float32)
    h_T = torch.tensor([0.85, 0.85, 0.20], dtype=torch.float32)
    
    burst_rate, mode = fused_dcn_rebound_burst(pc_rate, h_T, motor_threshold=150.0)
    
    # First neuron: Rebound burst
    assert mode[0].item() == DCNFiringMode.EXPLOSIVE_REBOUND
    assert burst_rate[0].item() >= 200.0
    
    # Second neuron: High Purkinje rate -> Tonic suppression
    assert mode[1].item() == DCNFiringMode.TONIC_SUPPRESSED
    assert burst_rate[1].item() <= 20.0
    
    # Third neuron: Low Purkinje but unprimed h_T -> Baseline spiking
    assert mode[2].item() == DCNFiringMode.BASELINE_SPIKING
    assert 30.0 <= burst_rate[2].item() <= 60.0

def test_dcn_avx2_binary_execution():
    bin_path = Path("build/weft-bench-dcn-rebound")
    assert bin_path.exists()
