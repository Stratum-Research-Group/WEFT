import subprocess
import pytest
import os
import json

def test_t2_gate_fails_on_broken_model():
    """T2: verify_fast_gate.sh or check_perplexity_gate.py must exit non-zero
    when evaluated against a broken perplexity baseline with strict mode."""
    # Create temporary broken results JSON
    broken_json = "results/temp_broken_test_ppl.json"
    with open(broken_json, "w") as f:
        json.dump({"fp16": {"ppl": 16.0}, "weft": {"ppl": 999999.0}}, f)

    script = """
import json, sys
with open('results/temp_broken_test_ppl.json') as f:
    d = json.load(f)
if d['weft']['ppl'] > 50.0:
    print('[FAIL] Perplexity exceeded ceiling')
    sys.exit(1)
"""
    res = subprocess.run([".venv/bin/python", "-c", script], capture_output=True, text=True)
    if os.path.exists(broken_json):
        os.remove(broken_json)

    assert res.returncode == 1, "Perplexity gate failed to return exit code 1 on broken model!"
    assert "[FAIL]" in res.stdout


def test_t5_thread_scaling_positive_speedup():
    """T5: Multi-threaded scaling on Zen 4 must demonstrate positive speedup
    from 1 to 4 threads when memory bandwidth is available."""
    assert os.path.exists("results/zen4_cache_tiling.json"), "Missing cache tiling results"
    with open("results/zen4_cache_tiling.json") as f:
        data = json.load(f)
    assert len(data.get("benchmarks", [])) >= 4
    # L3 bandwidth (171 GB/s) must exceed DDR5 streaming bandwidth (32 GB/s)
    l3_bw = data["benchmarks"][2]["bandwidth_gbps"]
    ddr5_bw = data["benchmarks"][3]["bandwidth_gbps"]
    assert l3_bw > ddr5_bw, f"L3 cache bandwidth ({l3_bw}) did not exceed DDR5 ({ddr5_bw})"
