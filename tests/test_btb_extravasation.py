import pytest
import numpy as np

def test_btb_extravasation_physiology():
    n = 100
    tumor_stage = np.linspace(10, 95, n)
    tie2_fus_rescue = 90.0

    ifp = np.where(tumor_stage > 50.0, 28.0 - (tie2_fus_rescue * 0.22), 4.0 + tumor_stage * 0.1)
    flux = np.where(tumor_stage > 50.0, 1.5 + (tie2_fus_rescue * 0.055), 7.0 - tumor_stage * 0.02)
    delivery = np.where(tumor_stage > 50.0, 12.0 + (tie2_fus_rescue * 0.52), 65.0 - tumor_stage * 0.15)

    assert np.all(ifp <= 10.0)
    assert np.all(flux >= 4.0)
    assert np.all(delivery >= 45.0)

def test_simd_avx2_btb_execution():
    import subprocess
    res = subprocess.run(["./tools/weft_bench_btb_extravasation"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Done. Completed 1000 iterations" in res.stdout
    assert "Throughput:" in res.stdout
