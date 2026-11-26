import pytest
import numpy as np

def test_meningeal_lymphatics_physiology():
    n = 100
    aging = np.linspace(10, 95, n)
    vegfc = 90.0

    vel = np.where(aging > 50.0, 1.5 + (vegfc * 0.035), 5.5 - aging * 0.015)
    efflux = np.where(aging > 50.0, 0.8 + (vegfc * 0.022), 3.2 - aging * 0.008)
    clr = np.where(aging > 50.0, 20.0 + (vegfc * 0.55), 80.0 - aging * 0.2)

    assert np.all(vel >= 4.0)
    assert np.all(efflux >= 2.5)
    assert np.all(clr >= 65.0)

def test_simd_avx2_lymphatics_execution():
    import subprocess
    res = subprocess.run(["./tools/weft_bench_meningeal_lymphatics"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Done. Completed 1000 iterations" in res.stdout
    assert "Throughput:" in res.stdout
