"""
Regression Test Suite: Target #403
Alzheimer's Disease: Corticocortical Gamma Uncoupling & PV Interneuron Hypoactivity
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group
"""

import pytest
import numpy as np
import torch

from weft.kernels.fused_alzheimer_gamma_triton import compute_alzheimer_gamma_triton


def test_alzheimer_healthy_baseline():
    """Verify that zero pathology load maintains fast-spiking PV rate (55 Hz), robust 40 Hz gamma, and steep 1/f slope."""
    path = torch.tensor([0.0], dtype=torch.float32)
    nav = torch.tensor([0.0], dtype=torch.float32)
    genus = torch.tensor([0.0], dtype=torch.float32)

    pv, gamma, chi, spikes = compute_alzheimer_gamma_triton(path, nav, genus)

    assert float(pv[0]) == pytest.approx(55.0, rel=1e-3)
    assert float(gamma[0]) == pytest.approx(50.0, rel=1e-3)
    assert float(chi[0]) == pytest.approx(2.80, rel=1e-3)
    assert float(spikes[0]) == pytest.approx(0.1, rel=1e-3)


def test_pathology_causes_gamma_loss_and_epileptiform_spikes():
    """Verify that 85% pathology load collapses PV rate < 25 Hz, drops gamma < 20 uV^2, and triggers spikes > 2.0 Hz."""
    path = torch.tensor([0.85], dtype=torch.float32)
    nav = torch.tensor([0.0], dtype=torch.float32)
    genus = torch.tensor([0.0], dtype=torch.float32)

    pv, gamma, chi, spikes = compute_alzheimer_gamma_triton(path, nav, genus)

    assert float(pv[0]) < 25.0
    assert float(gamma[0]) < 20.0
    assert float(chi[0]) < 1.60  # Flattened aperiodic spectral slope
    assert float(spikes[0]) > 2.0  # Subclinical epileptiform discharges


def test_pathology_monotonicity():
    """Verify that increasing amyloid/tau pathology monotonically reduces PV firing rate and flattens 1/f slope."""
    path_levels = torch.linspace(0.0, 1.0, 20)
    nav = torch.zeros(20, dtype=torch.float32)
    genus = torch.zeros(20, dtype=torch.float32)

    pv, gamma, chi, spikes = compute_alzheimer_gamma_triton(path_levels, nav, genus)
    pv_np = pv.cpu().numpy()

    # PV rate must decrease monotonically with increasing pathology
    diffs = np.diff(pv_np)
    assert np.all(diffs <= 1e-5), "PV firing rate must decrease monotonically as pathology increases"


def test_dual_nav11_genus_rescue():
    """Verify that combined Nav1.1 enhancer and GENUS 40 Hz entrainment restores gamma > 35 uV^2 and quenches spikes."""
    path = torch.tensor([0.85, 0.85], dtype=torch.float32)
    nav = torch.tensor([0.0, 0.85], dtype=torch.float32)
    genus = torch.tensor([0.0, 1.0], dtype=torch.float32)

    pv, gamma, chi, spikes = compute_alzheimer_gamma_triton(path, nav, genus)

    # Unrescued
    assert float(gamma[0]) < 20.0
    assert float(spikes[0]) > 2.0

    # Rescued
    assert float(gamma[1]) >= 35.0  # Gamma power restored
    assert float(chi[1]) >= 2.00  # 1/f slope steepened
    assert float(spikes[1]) < 0.5  # Epileptiform spikes quenched
    assert float(pv[1]) > float(pv[0])  # PV firing improved
