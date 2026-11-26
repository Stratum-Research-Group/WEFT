"""
Unit & Integration Test Suite for Reticulospinal Anticipatory Postural Adjustments (APAs)
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from weft.kernels.fused_reticulospinal_apa_triton import fused_reticulospinal_apa_step
from scripts.sim_reticulospinal_apa_reaching import run_reticulospinal_apa_sim

def test_reticulospinal_apa_simulation():
    res = run_reticulospinal_apa_sim()
    assert res["status"] == "success"
    assert res["feedforward_stabilization_verified"] is True
    assert abs(res["apa_lead_time_ms"] - 75.0) < 1e-3
    assert res["peak_com_displacement_mm"] < 5.0
    assert res["mean_apa_drive_hz"] > 120.0

def test_fused_reticulospinal_apa_kernel():
    B = 2
    N_S = 4

    # High intentional drive
    intent = torch.full((B, N_S), 80.0, dtype=torch.float32)

    # Phase 1: During APA lead window (-50 ms)
    t_apa = torch.full((B, N_S), -50.0, dtype=torch.float32)
    apa_1, focal_1, disrupt_1 = fused_reticulospinal_apa_step(intent, t_apa, apa_lead_ms=75.0)

    # APA active, focal inactive
    assert torch.all(apa_1 > 100.0)
    assert torch.all(focal_1 == 10.0)

    # Phase 2: During focal reaching (+50 ms)
    t_focal = torch.full((B, N_S), 50.0, dtype=torch.float32)
    apa_2, focal_2, disrupt_2 = fused_reticulospinal_apa_step(intent, t_focal, apa_lead_ms=75.0)

    # Both active and counter-balancing
    assert torch.all(apa_2 > 100.0)
    assert torch.all(focal_2 > 100.0)
    assert torch.all(disrupt_2 < 30.0)

def test_reticulospinal_avx2_binary():
    bin_path = Path("build/weft-bench-reticulospinal")
    assert bin_path.exists()
