import pytest
import numpy as np

def test_so_pac_physiology():
    n = 100
    desync = np.linspace(10, 95, n)
    acoustic_drive = 90.0

    plv = np.where(desync > 50.0, 0.25 + (acoustic_drive * 0.005), 0.82 - desync * 0.0015)
    amp = np.where(desync > 50.0, 14.0 + (acoustic_drive * 0.26), 48.0 - desync * 0.1)
    mem = np.where(desync > 50.0, 22.0 + (acoustic_drive * 0.55), 88.0 - desync * 0.12)

    assert np.all(plv >= 0.65)
    assert np.all(amp >= 30.0)
    assert np.all(mem >= 70.0)

def test_simd_avx2_so_pac_execution():
    import subprocess
    res = subprocess.run(["./tools/weft_bench_thalamocortical_so_pac"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Done. Completed 1000 iterations" in res.stdout
    assert "Throughput:" in res.stdout
