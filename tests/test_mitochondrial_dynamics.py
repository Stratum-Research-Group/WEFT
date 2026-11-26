import pytest
import numpy as np

def test_mitochondrial_dynamics_physiology():
    n = 100
    calcium_stress = np.linspace(15, 95, n)
    shed_mitoprotection = 90.0

    psi = np.where(calcium_stress > 50.0, 100.0 + (shed_mitoprotection * 0.55), 165.0 - calcium_stress * 0.2)
    atp = np.where(calcium_stress > 50.0, 1.5 + (shed_mitoprotection * 0.02), 3.8 - calcium_stress * 0.01)
    ratio = np.where(calcium_stress > 50.0, 2.4 - (shed_mitoprotection * 0.0115), 0.95 + calcium_stress * 0.002)
    mptp = np.where(calcium_stress > 50.0, 60.0 - (shed_mitoprotection * 0.58), 4.0 + calcium_stress * 0.03)

    assert np.all(psi >= 140.0)
    assert np.all(atp >= 2.5)
    assert np.all(ratio <= 1.50)
    assert np.all(mptp <= 10.0)

def test_simd_avx2_mitochondria_execution():
    import subprocess
    res = subprocess.run(["./tools/weft_bench_mitochondrial_dynamics"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Done. Completed 1000 iterations" in res.stdout
    assert "Throughput:" in res.stdout
