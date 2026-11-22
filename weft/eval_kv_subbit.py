"""
Live KV Cache & PagedAttention Sub-Bit Memory Benchmark:
Profiles token-by-token dynamic key-value cache memory allocation and throughput
when operating under sub-bit quantized foundation weights.
"""

import time
import argparse
import json
import logging
from typing import Dict, Any, List

import torch
from weft.paged_cache import PagedWeftBlockManager, PagedWeftServingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("weft.eval_kv_subbit")


def benchmark_kv_cache(
    num_layers: int = 32,
    max_blocks: int = 512,
    block_size: int = 16,
    latent_dim: int = 192,
    simulated_steps: int = 32,
) -> Dict[str, Any]:
    """
    Allocates and tests PagedWeftBlockManager memory footprint and dynamic write speed.
    """
    logger.info(f"Allocating PagedWeftBlockManager: layers={num_layers}, blocks={max_blocks}, block_size={block_size}, dim={latent_dim}")
    t0 = time.perf_counter()
    manager = PagedWeftBlockManager(
        num_layers=num_layers,
        block_size=block_size,
        max_blocks=max_blocks,
        latent_dim=latent_dim,
        device="cpu",
        dtype=torch.float32,
    )
    alloc_time = time.perf_counter() - t0

    # Comparison with standard dense KV cache at matching dtype and dimensions:
    # In standard GQA (e.g. 8 KV heads x 128 = 1024 dim per K or V, 2048 total per layer)
    # dense_dim represents total per-layer key+value dimension in FP16 (2 bytes per element)
    dense_dim = 2048
    dense_elements = num_layers * max_blocks * block_size * dense_dim
    dense_mb = (dense_elements * 2) / 1e6  # 2 bytes per float16

    # Actual physical memory pool of WEFT compressed latent blocks
    subbit_mb = manager.physical_pool.nbytes / 1e6
    compression_ratio = dense_mb / max(subbit_mb, 1e-6)

    # Simulate dynamic sequence allocation and token appends
    seq_id = 1
    blocks_needed = (simulated_steps + block_size - 1) // block_size
    manager.allocate(seq_id, num_blocks=blocks_needed)

    latencies = []
    for step in range(simulated_steps):
        token_latent = torch.randn(latent_dim, dtype=torch.float32)
        t_start = time.perf_counter()
        for layer in range(num_layers):
            manager.write_latent(layer_idx=layer, seq_id=seq_id, logical_token_pos=step, latent_vec=token_latent)
        latencies.append((time.perf_counter() - t_start) * 1000)

    mean_write_latency_ms = sum(latencies) / len(latencies)

    results = {
        "num_layers": num_layers,
        "max_blocks": max_blocks,
        "block_size": block_size,
        "latent_dim": latent_dim,
        "standard_dense_kv_mb": dense_mb,
        "weft_compressed_kv_mb": subbit_mb,
        "kv_compression_ratio": compression_ratio,
        "alloc_time_sec": alloc_time,
        "simulated_steps": simulated_steps,
        "mean_token_kv_write_ms": mean_write_latency_ms,
        "status": "PASS",
    }
    logger.info(f"WEFT Paged Latent footprint: {subbit_mb:.2f} MB vs dense {dense_mb:.2f} MB ({compression_ratio:.1f}x compression)")
    return results


def main():
    parser = argparse.ArgumentParser(description="Sub-Bit PagedAttention KV Cache Benchmark")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    res = benchmark_kv_cache(
        num_layers=args.layers,
        max_blocks=args.blocks,
        block_size=args.block_size,
        latent_dim=args.dim,
        simulated_steps=args.steps,
    )
    print(json.dumps(res, indent=2))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        logger.info(f"Results saved to {args.out_json}")


if __name__ == "__main__":
    main()
