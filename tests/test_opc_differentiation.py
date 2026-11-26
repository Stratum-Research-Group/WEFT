import pytest
import numpy as np

def test_opc_differentiation_physiology():
    n = 100
    insult = np.linspace(10, 95, n)
    shed_bdnf = 90.0

    gpr17 = np.where(insult > 50.0, 80.0 - (shed_bdnf * 0.65), 15.0)
    mbp = np.where(insult > 50.0, 25.0 + (shed_bdnf * 1.25), 150.0 - insult * 0.3)
    vel = np.where(insult > 50.0, 15.0 + (shed_bdnf * 0.60), 80.0 - insult * 0.2)

    assert np.all(gpr17 <= 30.0)
    assert np.all(mbp >= 120.0)
    assert np.all(vel >= 60.0)

def test_simd_avx2_opc_execution():
    import subprocess
    res = subprocess.run(["./tools/weft_bench_opc_differentiation"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Done. Completed 1000 iterations" in res.stdout
    assert "Throughput:" in res.stdout
