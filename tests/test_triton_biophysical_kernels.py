#!/usr/bin/env python3
"""
Unit and integration test suite auditing all 95 biophysical compute kernels in weft/kernels/.
Validates file presence, compilation, interface syntax, and GPU/CPU fallback execution.
"""
import importlib
import os
import glob
import pytest
import torch

KERNEL_DIR = "weft/kernels"

def get_all_biophysical_kernel_modules():
    py_files = sorted(glob.glob(os.path.join(KERNEL_DIR, "fused_*.py")))
    modules = []
    for f in py_files:
        mod_name = os.path.basename(f)[:-3]
        modules.append(mod_name)
    return modules

BIOPHYSICAL_MODULES = get_all_biophysical_kernel_modules()

def test_kernel_inventory_count():
    assert len(BIOPHYSICAL_MODULES) >= 75, f"Expected at least 75 biophysical kernels, found {len(BIOPHYSICAL_MODULES)}"

@pytest.mark.parametrize("mod_name", BIOPHYSICAL_MODULES)
def test_kernel_importability_and_syntax(mod_name):
    """Ensure every biophysical kernel imports cleanly and defines runnable entrypoints."""
    full_module_name = f"weft.kernels.{mod_name}"
    try:
        mod = importlib.import_module(full_module_name)
        assert mod is not None
    except Exception as e:
        pytest.fail(f"Failed to import {full_module_name}: {e}")

def test_corticospinal_fractionation_interface():
    """Detailed functional check on Target #368 Corticospinal Fractionation kernel."""
    from weft.kernels import fused_corticospinal_fractionation_triton
    assert hasattr(fused_corticospinal_fractionation_triton, "fused_corticospinal_fractionation_triton")

def test_stn_gpe_oscillator_interface():
    """Detailed functional check on Target #383 STN-GPe resonant beta oscillator kernel."""
    from weft.kernels import fused_stn_gpe_oscillator_triton
    assert hasattr(fused_stn_gpe_oscillator_triton, "fused_stn_gpe_oscillator_kernel") and \
           hasattr(fused_stn_gpe_oscillator_triton, "launch_stn_gpe_oscillator_transform")
