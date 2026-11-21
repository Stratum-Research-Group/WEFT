"""
Comparative Benchmark Across Sub-Bit Foundation Models:
Measures forward throughput, layerwise bits-per-weight, and compression ratios
across Qwen-2.5-72B, Llama-3.3-70B, Goedel-Prover-32B, and LiquidAI LFM3-VL-2B.
"""

from typing import Dict, Any, List
import os
import time
import torch
from weft.serve_any import load_universal_subbit_model


def run_frontier_subbit_benchmark() -> List[Dict[str, Any]]:
    models = [
        ("Qwen-2.5-72B", ["/run/media/akarta/AI/models/qwen-2.5-72b-subbit/model-00001-of-00037_subbit.weft"]),
        ("Llama-3.3-70B", ["/run/media/akarta/AI/models/llama-3.3-70b-subbit/model-00001-of-00030_subbit.weft"]),
        ("Goedel-Prover-32B", ["/run/media/akarta/AI/models/goedel-prover-32b-subbit/model-00001-of-00014_subbit.weft"]),
        ("LiquidAI-LFM3-VL-2B", ["/run/media/akarta/AI/models/lfm3-vl-2b-subbit/model_subbit.weft"]),
    ]

    results = []
    print("=" * 65)
    print("  WEFT Frontier Sub-Bit Multi-Model Comparative Benchmark")
    print("=" * 65)

    for name, shards in models:
        if not all(os.path.exists(s) for s in shards):
            continue
        model = load_universal_subbit_model(shards)
        summary = model.summary()

        # Benchmarking forward generation step
        dummy_prompt = torch.tensor([[1, 50, 100]])
        # Warmup
        _ = model.generate_step(dummy_prompt)

        t0 = time.perf_counter()
        iters = 10
        for _ in range(iters):
            _ = model.generate_step(dummy_prompt)
        dt = (time.perf_counter() - t0) / iters
        tok_per_sec = 1.0 / dt

        res = {
            "model": name,
            "layers_evaluated": summary["total_layers"],
            "parameters_m": summary["total_linear_parameters"] / 1e6,
            "effective_bpw": summary["effective_bpw"],
            "compression_ratio": summary["compression_ratio"],
            "latency_ms": dt * 1000.0,
            "tokens_per_sec": tok_per_sec,
        }
        results.append(res)
        print(f"[{name}]")
        print(f"  - Evaluated Layers: {res['layers_evaluated']}")
        print(f"  - Linear Parameters: {res['parameters_m']:.1f}M")
        print(f"  - Mean Bitrate: {res['effective_bpw']:.4f} BPW ({res['compression_ratio']:.1f}x)")
        print(f"  - Step Latency: {res['latency_ms']:.2f} ms ({res['tokens_per_sec']:.1f} tok/s)")
        print("-" * 65)

    return results


if __name__ == "__main__":
    run_frontier_subbit_benchmark()
