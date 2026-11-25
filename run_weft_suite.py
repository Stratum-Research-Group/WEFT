#!/usr/bin/env python3
"""
Master Test Runner for the WEFT Architecture Benchmark & Test Suite (v3).
Runs Track A (Architecture), Track B (Converter), and System Gates.
"""

import os
import sys
import argparse
import datetime
import json
import time

from weft_suite.common import get_environment_info
from weft_suite.tests_arch import (
    test_b1_causality,
    test_b2_prefill_decode_parity,
    test_b3_cache_formats,
    test_b5_exact_recurrence,
    test_b8_padding_and_batching
)
from weft_suite.tests_bench import (
    run_a1_language_modeling_sweep,
    run_a2_mqar_benchmark
)
from weft_suite.tests_conv import (
    test_c1_format_roundtrip,
    test_c5_architecture_labeling
)
from weft_suite.tests_regress import (
    test_rg1_no_synthetic_benchmark_claims,
    test_rg3_verify_fast_gate_has_real_checks,
    test_rg4_hardware_accuracy,
    test_rg7_repo_hygiene_no_root_tarballs
)

def main():
    parser = argparse.ArgumentParser(description="WEFT Benchmark & Test Suite Runner (v3)")
    parser.add_argument("--track", choices=["A", "B", "all"], default="all", help="Evaluation track")
    parser.add_argument("--tier", choices=["smoke", "standard", "full"], default="smoke", help="Execution tier")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Compute device")
    parser.add_argument("--out", default="benchmark_results", help="Output root directory")
    args = parser.parse_args()

    env_info = get_environment_info(args.device)
    run_id = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{env_info['git_commit_short']}"
    run_dir = os.path.join(args.out, run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    print("=" * 70)
    print("  WEFT ARCHITECTURE BENCHMARK & TEST SUITE (v3)")
    print(f"  Run ID:   {run_id}")
    print(f"  Tier:     {args.tier} | Track: {args.track} | Device: {args.device}")
    print(f"  Commit:   {env_info['git_commit_short']} (dirty={env_info['git_dirty']})")
    print("=" * 70)

    # Core gates
    tests_to_run = [
        ("B1_Causality", "Track A", "HARD", test_b1_causality, [args.device]),
        ("B2_Prefill_Decode_Parity", "Track A", "HARD", test_b2_prefill_decode_parity, [args.device]),
        ("B3_Cache_Formats", "Track A", "HARD", test_b3_cache_formats, [args.device]),
        ("B5_Exact_Recurrence", "Track A", "HARD", test_b5_exact_recurrence, [args.device]),
        ("B8_Padding_Batching", "Track A", "HARD", test_b8_padding_and_batching, [args.device]),
        ("C1_Format_Roundtrip", "Track B", "HARD", test_c1_format_roundtrip, []),
        ("C5_Arch_Labeling", "Track B", "HARD", test_c5_architecture_labeling, []),
        ("RG1_No_Synthetic_Claims", "Track B", "HARD", test_rg1_no_synthetic_benchmark_claims, []),
        ("RG3_Gate_Active_Checks", "System", "HARD", test_rg3_verify_fast_gate_has_real_checks, []),
        ("RG4_Hardware_Accuracy", "System", "HARD", test_rg4_hardware_accuracy, []),
        ("RG7_Repo_Hygiene", "System", "HARD", test_rg7_repo_hygiene_no_root_tarballs, []),
    ]

    results = []
    hard_failed = 0

    for test_id, track, gate_type, fn, fn_args in tests_to_run:
        t0 = time.time()
        try:
            ok, note = fn(*fn_args)
            duration = time.time() - t0
            status = "PASS" if ok else "FAIL"
            if not ok and gate_type == "HARD":
                hard_failed += 1
        except Exception as e:
            duration = time.time() - t0
            status = "FAIL"
            note = f"Exception: {str(e)}"
            if gate_type == "HARD":
                hard_failed += 1

        print(f"[{status:4s}] {test_id:<28s} ({track}, {gate_type}): {note}")

        test_record = {
            "test_id": test_id,
            "track": track,
            "gate_type": gate_type,
            "status": status,
            "duration_seconds": duration,
            "note": note
        }
        results.append(test_record)

        with open(os.path.join(run_dir, f"{test_id}.json"), "w", encoding="utf-8") as f:
            json.dump(test_record, f, indent=2)

    # If tier is standard or full, run benchmarks
    if args.tier in ["standard", "full"]:
        print("\n--- Running Track A Benchmarks ---")
        steps = 100 if args.tier == "standard" else 500
        a1_res = run_a1_language_modeling_sweep(args.device, steps=steps)
        with open(os.path.join(run_dir, "A1_LM_Sweep.json"), "w", encoding="utf-8") as f:
            json.dump(a1_res, f, indent=2)

        a2_res = run_a2_mqar_benchmark(args.device)
        with open(os.path.join(run_dir, "A2_MQAR.json"), "w", encoding="utf-8") as f:
            json.dump(a2_res, f, indent=2)

    # Generate SUMMARY.md
    summary_path = os.path.join(run_dir, "SUMMARY.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# WEFT Architecture Suite Execution Summary\n\n")
        f.write(f"- **Run ID:** `{run_id}`\n")
        f.write(f"- **Timestamp (UTC):** `{env_info['timestamp_utc']}`\n")
        f.write(f"- **Git Commit:** `{env_info['git_commit']}` (dirty={env_info['git_dirty']})\n")
        f.write(f"- **Tier:** `{args.tier}` | **Track:** `{args.track}` | **Device:** `{args.device}`\n")
        f.write(f"- **Verdict:** **{'ALL GATES PASSED' if hard_failed == 0 else 'HARD GATES FAILED'}**\n\n")
        f.write("| Test ID | Track | Gate Type | Status | Duration (s) | Notes |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            f.write(f"| `{r['test_id']}` | {r['track']} | {r['gate_type']} | **{r['status']}** | {r['duration_seconds']:.2f}s | {r['note']} |\n")

    print("\n" + "=" * 70)
    print(f"  SUITE RUN COMPLETE. Verdict: {'ALL GATES PASSED' if hard_failed == 0 else 'HARD GATES FAILED'}")
    print(f"  Summary saved to {summary_path}")
    print("=" * 70)

    sys.exit(0 if hard_failed == 0 else 1)

if __name__ == "__main__":
    main()
