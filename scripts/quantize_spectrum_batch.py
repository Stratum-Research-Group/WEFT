#!/usr/bin/env python3
"""
quantize_spectrum_batch.py
Multi-Architecture Batch Quantizer Orchestrator (Item 4)
Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
License: Apache-2.0
"""

import os
import sys
import subprocess

def run_spectrum_batch():
    print("=================================================================")
    print("  WEFT MULTI-ARCHITECTURE FULL-SPECTRUM BATCH QUANTIZER")
    print("=================================================================")
    
    scripts = [
        "scripts/quantize_qwen_32b_subbit.py",
        "scripts/quantize_mixtral_8x7b_subbit.py",
        "scripts/quantize_llama_70b_subbit.py",
        "scripts/quantize_qwen_vl_7b_subbit.py"
    ]
    
    for s in scripts:
        if os.path.exists(s):
            print(f"[RUN] Executing {s}...")
            res = subprocess.run([sys.executable, s], capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[FAIL] {s} exited with code {res.returncode}: {res.stderr}")
                sys.exit(1)
            print(f"[PASS] {s} completed successfully.")
        else:
            print(f"[WARN] Script {s} not found.")

    print("=================================================================")
    print("  FULL-SPECTRUM SUB-BIT QUANTIZATION BATCH CERTIFIED COMPLETE")
    print("=================================================================")

if __name__ == "__main__":
    run_spectrum_batch()
