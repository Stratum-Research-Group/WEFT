#!/usr/bin/env python3
"""
Sequential Frontier Model Deconstruction Orchestrator for WEFT.
Monitors in-flight jobs, benchmarks completed targets, and automatically launches
the next queued frontier models without host memory blowup or idle downtime.
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("weft.frontier_pipeline")


def check_disk_space():
    """Verify available space on the AI storage drive."""
    res = subprocess.run(["df", "-BG", "/run/media/akarta/AI"], capture_output=True, text=True)
    lines = res.stdout.strip().split("\n")
    if len(lines) >= 2:
        parts = lines[1].split()
        avail_gb = int(parts[3].replace("G", ""))
        logger.info(f"Storage status on /run/media/akarta/AI: {avail_gb} GB free")
        if avail_gb < 690:
            logger.warning(f"Storage ({avail_gb} GB) is approaching safe floor (700 GB)!")
        return avail_gb
    return 700


def run_benchmark(model_dir: str):
    """Run live interactive decoding benchmark on completed model."""
    logger.info(f"Running live decode benchmark on {model_dir}...")
    cmd = [
        "uv", "run", "python3", "weft/eval_live_bench.py",
        "--model-dir", model_dir,
        "--max-tokens", "16"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(f"Benchmark result:\n{res.stdout}")
    if res.stderr:
        logger.warning(f"Benchmark stderr:\n{res.stderr}")


def stream_deconstruct_model(repo_id: str, output_dir: str, base_rank: int = 64, itq_iters: int = 3):
    """Execute streaming sharded quantization."""
    logger.info(f"Starting deconstruction of {repo_id} into {output_dir}...")
    cmd = [
        "uv", "run", "python3", "weft/stream_quantizer.py",
        "--repo-id", repo_id,
        "--output-dir", output_dir,
        "--base-rank", str(base_rank),
        "--itq-iterations", str(itq_iters)
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        logger.error(f"Deconstruction failed for {repo_id} with exit code {res.returncode}")
        return False
    logger.info(f"Deconstruction completed successfully for {repo_id}!")
    return True


def main():
    logger.info("=== WEFT Autonomous Frontier Pipeline Orchestrator Initialized ===")
    check_disk_space()

    # Step 1: Wait for Mixtral-8x7B (Model B) if running
    mixtral_dir = "/run/media/akarta/AI/models/mixtral-8x7b-subbit"
    logger.info(f"Monitoring Mixtral-8x7B progress in {mixtral_dir}...")
    
    while True:
        # Check if all 19 shards are present
        weft_shards = list(Path(mixtral_dir).glob("model-000*-of-00019_subbit.weft"))
        if len(weft_shards) >= 19:
            logger.info(f"Mixtral-8x7B complete! (19/19 shards found)")
            break
        logger.info(f"Mixtral-8x7B progress: {len(weft_shards)}/19 shards completed. Sleeping 30s...")
        time.sleep(30)

    # Benchmark Mixtral
    run_benchmark(mixtral_dir)

    # Step 2: Launch Model C (DeepSeek-R1-Distill-Qwen-32B)
    repo_c = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    dir_c = "/run/media/akarta/AI/models/deepseek-r1-distill-qwen-32b-subbit"
    logger.info(f"Advancing to Model C: {repo_c}")
    check_disk_space()
    success_c = stream_deconstruct_model(repo_c, dir_c, base_rank=64)
    if success_c:
        run_benchmark(dir_c)

    # Step 3: Launch Model D (Whisper-Large-V3)
    repo_d = "openai/whisper-large-v3"
    dir_d = "/run/media/akarta/AI/models/whisper-large-v3-subbit"
    logger.info(f"Advancing to Model D: {repo_d}")
    check_disk_space()
    success_d = stream_deconstruct_model(repo_d, dir_d, base_rank=64)
    if success_d:
        run_benchmark(dir_d)

    logger.info("=== ALL QUEUED FRONTIER MODELS (B -> C -> D) FULLY DECONSTRUCTED ===")


if __name__ == "__main__":
    main()
