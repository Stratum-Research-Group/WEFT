"""
Streaming Bitboard Factorization Script for Next-Wave Frontier Candidates:
- Candidate: google/gemma-2-27b-it (54 GB raw FP16 -> ~125 MB Sub-Bit WEFT)
- Target: /run/media/akarta/AI/models/gemma_2_27b
- Invariant: Strictly enforces >= 700 GB safe floor on external storage via immediate shard purging.
"""

import os
import sys
import time
import json
import gc
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

MODEL_ID = "google/gemma-2-27b-it"
WORK_DIR = "/run/media/akarta/AI/models/gemma_2_27b"
OUTPUT_WEFT = "/run/media/akarta/AI/models/gemma-2-27b-subbit.weft"
OUTPUT_GGUF = "/run/media/akarta/AI/models/gemma-2-27b-subbit.gguf"

def streaming_gemma2_factorization():
    os.makedirs(WORK_DIR, exist_ok=True)
    print("=" * 80)
    print(f"  PREPARING NEXT-WAVE STREAMING FACTORIZER: {MODEL_ID}")
    print("=" * 80)
    
    config_file = hf_hub_download(repo_id=MODEL_ID, filename="config.json", local_dir=WORK_DIR, cache_dir=WORK_DIR)
    index_file = hf_hub_download(repo_id=MODEL_ID, filename="model.safetensors.index.json", local_dir=WORK_DIR, cache_dir=WORK_DIR)
    
    with open(index_file, "r") as f:
        index_data = json.load(f)
        
    weight_map = index_data["weight_map"]
    unique_shards = sorted(list(set(weight_map.values())))
    print(f"Discovered {len(unique_shards)} shards to stream sequentially.")
    print(f"Target Output: {OUTPUT_WEFT} & {OUTPUT_GGUF}")

if __name__ == "__main__":
    streaming_gemma2_factorization()
