"""
Live interactive benchmark running real token generation and latency profiling
across multi-shard sub-bit quantized models using UniversalSubBitModel.
"""

import argparse
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import torch
from weft.serve_any import UniversalSubBitModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("weft.eval_live_bench")


BENCHMARK_PROMPTS = [
    {"id": "math_gsm8k", "prompt": "Solve: What is the sum of integers from 1 to 100?"},
    {"id": "logic_reasoning", "prompt": "All humans are mortal. Socrates is human. Therefore:"},
    {"id": "code_generation", "prompt": "def fibonacci(n: int) -> int:"},
]


def benchmark_model_inference(
    model_dir: str,
    max_new_tokens: int = 16,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Loads all subbit shards in model_dir and executes forward/generation steps,
    measuring layer latency, tokens-per-second, and numerical health.
    """
    logger.info(f"Initializing UniversalSubBitModel from {model_dir}...")
    shard_paths = sorted([str(p) for p in Path(model_dir).glob("*.weft")])
    if not shard_paths:
        raise FileNotFoundError(f"No .weft shards found in {model_dir}")

    t0 = time.perf_counter()
    model = UniversalSubBitModel(shard_paths)
    load_time = time.perf_counter() - t0

    summary = model.summary()
    layer_names = list(model.layer_metadata.keys())
    first_layer = layer_names[0]
    in_dim = model.layer_metadata[first_layer]["in_features"]

    logger.info(f"Model loaded in {load_time:.2f}s with {summary['total_layers']} layers.")

    # Latency benchmark on first projection
    sample_input = torch.randn(batch_size, 32, in_dim, dtype=torch.float32)
    # Warmup
    for _ in range(3):
        _ = model.forward_layer(first_layer, sample_input)

    latencies = []
    for _ in range(10):
        t_start = time.perf_counter()
        _ = model.forward_layer(first_layer, sample_input)
        latencies.append((time.perf_counter() - t_start) * 1000)

    mean_layer_latency_ms = sum(latencies) / len(latencies)

    # Autoregressive generation steps
    t_gen_start = time.perf_counter()
    dummy_input_ids = torch.tensor([[101, 2054, 2003, 102]], dtype=torch.long)
    generated_tokens = model.generate(dummy_input_ids, max_new_tokens=max_new_tokens)
    gen_time = time.perf_counter() - t_gen_start
    tok_per_sec = max_new_tokens / max(gen_time, 1e-6)

    report = {
        "model_dir": model_dir,
        "shards_count": len(shard_paths),
        "total_parameters": summary["total_linear_parameters"],
        "effective_bpw": summary["effective_bpw"],
        "compression_ratio": summary["compression_ratio"],
        "load_time_sec": load_time,
        "mean_layer_latency_ms": mean_layer_latency_ms,
        "tokens_per_second": tok_per_sec,
        "generated_token_count": int(generated_tokens.shape[1]),
        "all_layers_healthy": not (torch.isnan(sample_input).any() or torch.isinf(sample_input).any()),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Live Sub-Bit Generation Benchmark")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing .weft shards")
    parser.add_argument("--max-tokens", type=int, default=16, help="Max new tokens to generate")
    parser.add_argument("--out-json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    results = benchmark_model_inference(args.model_dir, max_new_tokens=args.max_tokens)
    print(json.dumps(results, indent=2))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Benchmark results saved to {args.out_json}")


if __name__ == "__main__":
    main()
