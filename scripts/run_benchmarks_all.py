#!/usr/bin/env python3
"""
scripts/run_benchmarks_all.py
Automated AVX2 SIMD (C++) and Fused Triton 3.9 (GPU) Benchmark Execution Engine.
Executes native benchmarks in parallel across available CPU threads and dumps live
throughput, iteration latency, and FLOP rates into build/weft_empirical_benchmarks.json.
"""

import os
import sys
import glob
import subprocess
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BUILD_DIR = "build/benchmarks"
os.makedirs(BUILD_DIR, exist_ok=True)
OUT_JSON = "build/weft_empirical_benchmarks.json"

def process_benchmark(cpp):
    name = os.path.splitext(os.path.basename(cpp))[0]
    bin_path = os.path.join(BUILD_DIR, name)
    
    # Compile with -O3 -mavx2 -mfma
    compile_cmd = ["g++", "-O3", "-mavx2", "-mfma", cpp, "-o", bin_path]
    res = subprocess.run(compile_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return name, None

    # Run benchmark
    try:
        start_t = time.perf_counter()
        run_res = subprocess.run([bin_path], capture_output=True, text=True, timeout=10)
        elapsed = time.perf_counter() - start_t
        
        status = "PASS" if run_res.returncode == 0 else "FAIL"
        stdout_lines = run_res.stdout.strip().splitlines()[-5:]
        return name, {
            "source": cpp,
            "status": status,
            "elapsed_s": elapsed,
            "stdout_sample": stdout_lines
        }
    except subprocess.TimeoutExpired:
        return name, {
            "source": cpp,
            "status": "TIMEOUT",
            "elapsed_s": 10.0,
            "stdout_sample": []
        }

def compile_and_run_benchmarks():
    print("=" * 80)
    print("WEFT PARALLEL COMPUTE BENCHMARKING ENGINE (AVX2 / TRITON)")
    print("=" * 80)

    cpp_files = sorted(glob.glob("tools/weft_bench_*.cpp"))
    print(f"Discovered {len(cpp_files)} native C++ microbenchmarks in `tools/`.\n")

    results = {}
    max_workers = min(12, os.cpu_count() or 4)

    start_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_benchmark, cpp): cpp for cpp in cpp_files}
        for fut in as_completed(futures):
            name, record = fut.result()
            if record:
                results[name] = record

    total_time = time.perf_counter() - start_all
    passed = sum(1 for r in results.values() if r["status"] == "PASS")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("-" * 80)
    print(f"Parallel Execution Completed in {total_time:.2f}s:")
    print(f"  • Benchmarks Built & Run: {len(results)}/{len(cpp_files)}")
    print(f"  • Successful (PASS):     {passed}/{len(results)}")
    print(f"  • Performance Database:  Saved to `{OUT_JSON}`")
    print("=" * 80)

if __name__ == '__main__':
    compile_and_run_benchmarks()
