"""
End-to-End Streaming Benchmark for 1.04T MoE Sharded Safetensors-to-WEFT Conversion on NVMe.
Measures throughput (MB/s and GB/s), latency per shard, and verifies storage safety floor invariant.
"""

import os
import sys
import time
import json
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from weft.convert_llama_405b_streaming import verify_storage_safety
from scripts.deconstruct_frontier_1t_moe import stream_deconstruct_frontier_1t

def benchmark_1t_streaming():
    ai_free_start, home_free_start = verify_storage_safety()
    process = psutil.Process()
    start_rss = process.memory_info().rss / (1024**2)

    num_shards = 16
    t0 = time.perf_counter()
    manifest = stream_deconstruct_frontier_1t(num_shards=num_shards)
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    end_rss = process.memory_info().rss / (1024**2)
    ai_free_end, home_free_end = verify_storage_safety()
    
    total_kb = sum(s["size_kb"] for s in manifest["shards"])
    throughput_mb_s = (total_kb / 1024.0) / elapsed

    results = {
        "model": "weft-frontier-1.04T-moe",
        "num_shards": num_shards,
        "elapsed_seconds": elapsed,
        "latency_per_shard_ms": (elapsed / num_shards) * 1000.0,
        "total_converted_mb": total_kb / 1024.0,
        "throughput_mb_s": throughput_mb_s,
        "start_rss_mb": start_rss,
        "peak_rss_mb": end_rss,
        "storage_ai_start_gb": ai_free_start,
        "storage_ai_end_gb": ai_free_end,
        "storage_floor_preserved": ai_free_end >= 700.0 and home_free_end >= 198.0
    }

    os.makedirs("results", exist_ok=True)
    out_file = "results/frontier_1t_streaming_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("=================================================================")
    print("  FRONTIER 1.04T STREAMING DECONSTRUCTION BENCHMARK RESULTS")
    print("=================================================================")
    print(f"Elapsed Time       : {elapsed:.2f} s")
    print(f"Latency / Shard    : {results['latency_per_shard_ms']:.2f} ms")
    print(f"Throughput         : {throughput_mb_s:.2f} MB/s")
    print(f"Peak RSS Memory    : {end_rss:.2f} MB (< 2000 MB target)")
    print(f"Floor Preserved    : {results['storage_floor_preserved']} (AI: {ai_free_end:.2f} GB)")
    print(f"Committed Artifact : {out_file}")
    print("=================================================================")
    return results

if __name__ == "__main__":
    benchmark_1t_streaming()
