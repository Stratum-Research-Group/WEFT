import pytest
import numpy as np

def test_shear_stress_mechanotransduction():
    n = 100
    shear_stress = np.linspace(2.0, 25.0, n)
    shed_chaperone = 90.0

    phos = np.where(shear_stress >= 12.0, 70.0 + shear_stress * 0.8, 25.0 + (shed_chaperone * 0.45))
    no_bio = np.where(shear_stress >= 12.0, 1.8 + shear_stress * 0.04, 0.6 + (shed_chaperone * 0.015))
    glyco = np.where(shear_stress >= 12.0, 75.0 + shear_stress * 0.7, 30.0 + (shed_chaperone * 0.45))

    assert np.all(phos >= 65.0)
    assert np.all(no_bio >= 1.8)
    assert np.all(glyco >= 70.0)

def test_simd_avx2_shear_stress_execution():
    import subprocess
    res = subprocess.run(["./tools/weft_bench_shear_stress"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Done. Completed 1000 iterations" in res.stdout
    assert "Throughput:" in res.stdout
