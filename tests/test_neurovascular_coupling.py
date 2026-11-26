import pytest
import numpy as np

def test_neurovascular_coupling_physiology():
    n = 100
    baseline_radius = 20.0
    glutamate_stim = np.linspace(20, 80, n)
    endfoot_ca = 120.0 + glutamate_stim * 3.5
    perivascular_k = 3.0 + (endfoot_ca / 500.0) * 10.0
    eet = glutamate_stim * 0.8
    delta_r = (perivascular_k - 3.0) * 0.8 + eet * 0.05
    radius = baseline_radius + delta_r
    flux = (radius / baseline_radius) ** 4

    assert np.all(endfoot_ca >= 180.0)
    assert np.all(endfoot_ca <= 450.0)
    assert np.all(perivascular_k >= 6.0)
    assert np.all(perivascular_k <= 14.0)
    assert np.all(radius > baseline_radius)
    assert np.all(flux >= 1.0)

def test_simd_avx2_binary_execution():
    import subprocess
    res = subprocess.run(["./tools/weft_bench_neurovascular_coupling"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Done. Completed 1000 iterations" in res.stdout
    assert "Throughput:" in res.stdout
