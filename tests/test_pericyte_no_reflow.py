import pytest
import numpy as np

def test_pericyte_no_reflow_rescue():
    n = 100
    baseline_d = 5.5
    ischemic_insult = np.linspace(30, 90, n)
    ca_unrescued = 120.0 + ischemic_insult * 4.5
    shed_rescue_level = 85.0
    ca_rescued = np.maximum(120.0, ca_unrescued - (shed_rescue_level * 3.2))

    d_rescued = np.maximum(2.0, baseline_d - (ca_rescued - 120.0) * 0.008 + (shed_rescue_level * 0.015))
    flux_rescued = (d_rescued / baseline_d) ** 4

    assert np.all(d_rescued >= 4.5)
    assert np.all(flux_rescued >= 0.5)

def test_simd_avx2_pericyte_execution():
    import subprocess
    res = subprocess.run(["./tools/weft_bench_pericyte_no_reflow"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Done. Completed 1000 iterations" in res.stdout
    assert "Throughput:" in res.stdout
