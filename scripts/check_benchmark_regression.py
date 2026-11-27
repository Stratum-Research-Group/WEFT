#!/usr/bin/env python3
"""
Performance & Numerical Regression Gatekeeper:
Audits benchmark output reports against strict golden threshold criteria.
Fails CI if throughput degrades or numerical fidelity drops.
"""

import json
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "benchmarks_reports"
TRITON_REPORT = REPORTS_DIR / "TRITON_BIOPHYSICAL_BENCHMARK_REPORT.json"
QUANT_REPORT = REPORTS_DIR / "QUANTIZATION_DRIFT_EVALUATION.json"

def check_regressions():
    print("=== WEFT Automated Performance & Numerical Regression Gate ===")
    failures = []

    # 1. Audit Triton Biophysical Kernels
    if not TRITON_REPORT.exists():
        failures.append(f"Missing Triton benchmark report: {TRITON_REPORT}")
    else:
        triton_data = json.loads(TRITON_REPORT.read_text(encoding="utf-8"))
        kernels = triton_data.get("kernels", {})
        if not kernels:
            failures.append("No benchmarked kernels recorded in Triton report.")
        for k_name, k_info in kernels.items():
            if k_info.get("status") != "PASS":
                failures.append(f"Triton kernel {k_name} failed execution.")
            lat = k_info.get("latency_ms", 999.0)
            if lat > 500.0:
                failures.append(f"Latency regression in {k_name}: {lat} ms > 500 ms threshold.")
        print(f"✓ Triton Biophysical Suite: {len(kernels)} kernels verified within latency bounds.")

    # 2. Audit Quantization Precision Sensitivity
    if not QUANT_REPORT.exists():
        failures.append(f"Missing Quantization evaluation report: {QUANT_REPORT}")
    else:
        quant_data = json.loads(QUANT_REPORT.read_text(encoding="utf-8"))
        cos_retention = quant_data.get("cosine_retention_pct", 0.0)
        # Threshold: Sub-bit recurrent state must maintain >= 90.0% cosine fidelity to continuous FP32
        if cos_retention < 90.0:
            failures.append(f"Quantization numerical drift: Cosine retention {cos_retention}% < 90.0% threshold.")
        else:
            print(f"✓ Quantization Fidelity: Sub-bit cosine retention = {cos_retention}% (>= 90.0% threshold).")

    if failures:
        print("\n❌ REGRESSION GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("\n✅ REGRESSION GATE PASSED: All latency and numerical invariants satisfied.")
        return 0

if __name__ == "__main__":
    sys.exit(check_regressions())
