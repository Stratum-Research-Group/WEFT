"""
WEFT-2 Overnight Autonomous Execution Pipeline (7-Hour Horizon).

Orchestrates sequential, crash-safe stages:
1. Warm-starts policy from newly pretrained 50k foundation weights (checkpoints/weft2_extended_50k/final).
2. Executes 200-Episode Scaled GRPO / RLVR alignment with strict step-by-step reasoning rewards.
3. Automatically quantizes the aligned policy into certified sub-bit .weft (LittleBit-2 Joint-ITQ).
4. Solves Lyapunov interval certificate (all layers bounded < 1.0).
5. Exports consolidated GGUF container.
6. Runs full end-to-end live inference benchmark suite across math, logic, and tool trajectories.
"""

import os
import sys
import time
import json
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from examples.train_weft2_tool_grpo_scaled import run_scaled_tool_grpo
from weft.converter import convert_model_to_weft
from weft.export_subbit_gguf import export_weft_to_gguf
from weft.eval_live_bench import benchmark_model_inference
import tempfile
import shutil

BASE_50K = "checkpoints/weft2_extended_50k/final"
RLVR_OUT = "checkpoints/weft2_extended_50k_rlvr"
WEFT_OUT = "checkpoints/weft2_extended_50k_rlvr.weft"
GGUF_OUT = "checkpoints/weft2_extended_50k_rlvr.gguf"
BENCH_OUT = "benchmarks/results/weft2_extended_50k_rlvr_eval.json"

print("=" * 80)
print("  WEFT-2 AUTONOMOUS OVERNIGHT EXECUTION PIPELINE")
print("=" * 80)
print(f"Base Checkpoint: {BASE_50K}")
print(f"RLVR Policy Output: {RLVR_OUT}")
print(f"Sub-Bit Target: {WEFT_OUT} / {GGUF_OUT}")
print(f"Benchmark Output: {BENCH_OUT}")
print("=" * 80)

t_start = time.time()

# Stage 1: Run 200-Episode Scaled Tool GRPO Policy Optimization
print("\n>>> STAGE 1/4: Launching 200-Episode Scaled RLVR Policy Optimization on CUDA...")
run_scaled_tool_grpo(model_path=BASE_50K, save_path=RLVR_OUT, episodes=200)

# Stage 2: Sub-Bit Joint-ITQ Factorization & Lyapunov Interval Certification
print("\n>>> STAGE 2/4: Sub-Bit Quantization & Mathematical Interval Certification...")
with open(os.path.join(RLVR_OUT, "config.json")) as f:
    config = json.load(f)

state_dict = {}
with safe_open(os.path.join(RLVR_OUT, "model.safetensors"), framework="pt") as f:
    for k in f.keys():
        state_dict[k] = f.get_tensor(k)

container, cert = convert_model_to_weft(
    state_dict=state_dict,
    config=config,
    output_filepath=WEFT_OUT,
    base_rank=32,
    adaptive_rank=True,
    certify_intervals=True,
)
print(f"Successfully generated {WEFT_OUT} ({os.path.getsize(WEFT_OUT)/1024**2:.2f} MB) with {len(cert)} certified layers.")

# Stage 3: Export Consolidated GGUF
print("\n>>> STAGE 3/4: Exporting Consolidated GGUF Container...")
export_weft_to_gguf(WEFT_OUT, GGUF_OUT)
print(f"Successfully exported {GGUF_OUT} ({os.path.getsize(GGUF_OUT)/1024**2:.2f} MB).")

# Stage 4: Live Generation & Numerical Health Benchmark
print("\n>>> STAGE 4/4: Executing Live Multi-Domain Generation Benchmark...")
with tempfile.TemporaryDirectory() as tmpdir:
    shutil.copy(WEFT_OUT, os.path.join(tmpdir, "model.weft"))
    bench_results = benchmark_model_inference(tmpdir, max_new_tokens=24)
    os.makedirs(os.path.dirname(BENCH_OUT), exist_ok=True)
    with open(BENCH_OUT, "w") as f:
        json.dump(bench_results, f, indent=2)

print("\n" + "=" * 80)
print(f"ALL OVERNIGHT STAGES COMPLETED IN {(time.time() - t_start)/60:.1f} MINUTES.")
print("=" * 80)
print(json.dumps(bench_results, indent=2))
