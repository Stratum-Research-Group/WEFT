"""
tests/test_vagal_anti_inflammatory.py
Regression tests for Vagal Cholinergic Anti-Inflammatory Reflex models.
"""

import pytest
import torch
from weft.kernels.fused_vagal_anti_inflammatory_triton import fused_vagal_anti_inflammatory_triton


def test_vagal_anti_inflammatory_triton_cpu_fallback():
    freq = torch.tensor([0.0, 10.0, 20.0, 30.0], dtype=torch.float32)
    lps = torch.tensor([80.0, 80.0, 80.0, 80.0], dtype=torch.float32)

    ach, occ, stat3, nfkb, tnf = fused_vagal_anti_inflammatory_triton(freq, lps)

    # Vagus = 0 Hz: no ACh, no occupancy, high NF-kB
    assert ach[0].item() == 0.0
    assert occ[0].item() == 0.0
    assert nfkb[0].item() == 80.0
    assert tnf[0].item() == pytest.approx(15.0 + 80.0 * 1.4, rel=1e-3)

    # Vagus = 20 Hz: significant ACh, >60% occupancy, suppressed NF-kB
    assert ach[2].item() > 400.0
    assert occ[2].item() >= 60.0
    assert stat3[2].item() > 2.5
    assert nfkb[2].item() <= 25.0
    assert tnf[2].item() <= 55.0


def test_vagal_anti_inflammatory_monotonicity():
    freqs = torch.linspace(0.0, 35.0, 36)
    lps = torch.full_like(freqs, 85.0)

    ach, occ, stat3, nfkb, tnf = fused_vagal_anti_inflammatory_triton(freqs, lps)

    # Occupancy must be monotonically non-decreasing
    diff_occ = torch.diff(occ)
    assert torch.all(diff_occ >= -1e-5), "Receptor occupancy is not monotonically non-decreasing"

    # NF-kB translocation must be monotonically non-increasing
    diff_nfkb = torch.diff(nfkb)
    assert torch.all(diff_nfkb <= 1e-5), "Nuclear NF-kB translocation is not monotonically non-increasing"
