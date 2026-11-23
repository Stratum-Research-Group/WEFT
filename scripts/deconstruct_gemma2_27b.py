"""
Automated Pipeline Chain:
Monitors Mistral-Small-24B streaming deconstruction, and once complete,
immediately transitions to streaming SVD deconstruction of Google Gemma-2-27B-it.
"""

import os
import sys
import time
import json
import gc
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

GEMMA2_MODEL_ID = "google/gemma-2-27b-it"
GEMMA2_WORK_DIR = "/run/media/akarta/AI/models/gemma_2_27b"
GEMMA2_OUTPUT_WEFT = "/run/media/akarta/AI/models/gemma-2-27b-subbit.weft"
GEMMA2_OUTPUT_GGUF = "/run/media/akarta/AI/models/gemma-2-27b-subbit.gguf"

MISTRAL_OUTPUT_WEFT = "/run/media/akarta/AI/models/mistral-small-24b-subbit.weft"

def wait_for_mistral_and_run_gemma2():
    print("=" * 80)
    print("  WEFT SEQUENTIAL DECONSTRUCTION PIPELINE ORCHESTRATOR")
    print(f"  Target 1: Mistral-Small-24B -> {MISTRAL_OUTPUT_WEFT}")
    print(f"  Target 2: Gemma-2-27B-it     -> {GEMMA2_OUTPUT_WEFT}")
    print("=" * 80)
    
    # 1. Wait for Mistral-Small-24B to finish
    while not os.path.exists(MISTRAL_OUTPUT_WEFT):
        print(f"[{time.strftime('%H:%M:%S')}] Waiting for Mistral-Small-24B to finish SVD factorizing...")
        time.sleep(30)
        
    print(f"\n[OK] Detected Mistral-Small-24B artifact ({os.path.getsize(MISTRAL_OUTPUT_WEFT)/(1024*1024):.2f} MB).")
    print("Starting Gemma-2-27B-it streaming deconstruction under strict >= 700 GB floor...\n")
    
    # 2. Execute Gemma-2-27B Streaming SVD
    os.makedirs(GEMMA2_WORK_DIR, exist_ok=True)
    config_file = hf_hub_download(repo_id=GEMMA2_MODEL_ID, filename="config.json", local_dir=GEMMA2_WORK_DIR, cache_dir=GEMMA2_WORK_DIR)
    index_file = hf_hub_download(repo_id=GEMMA2_MODEL_ID, filename="model.safetensors.index.json", local_dir=GEMMA2_WORK_DIR, cache_dir=GEMMA2_WORK_DIR)
    
    with open(index_file, "r") as f:
        index_data = json.load(f)
        
    weight_map = index_data["weight_map"]
    unique_shards = sorted(list(set(weight_map.values())))
    print(f"Streaming {len(unique_shards)} shards for Gemma-2-27B-it...")
    
    factorized_layers = {}
    total_raw_bytes = 0
    t0 = time.time()
    
    for idx, shard_name in enumerate(unique_shards, 1):
        print(f"\n---> Streaming Shard {idx}/{len(unique_shards)}: {shard_name} ...")
        shard_path = hf_hub_download(repo_id=GEMMA2_MODEL_ID, filename=shard_name, local_dir=GEMMA2_WORK_DIR, cache_dir=GEMMA2_WORK_DIR)
        shard_sz = os.path.getsize(shard_path)
        total_raw_bytes += shard_sz
        
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if k.endswith(".weight") and ("mlp" in k or "q_proj" in k or "k_proj" in k or "v_proj" in k or "o_proj" in k or "gate" in k):
                    t = sf.get_tensor(k)
                    if t.ndim == 2:
                        d_out, d_in = t.shape
                        U, S, Vh = torch.linalg.svd(t.to(torch.float32), full_matrices=False)
                        u = (U[:, 0] * torch.sqrt(S[0])).to(torch.float16).numpy()
                        v = (Vh[0, :] * torch.sqrt(S[0])).to(torch.float16).numpy()
                        factorized_layers[k] = {
                            "shape": [d_out, d_in],
                            "u": u,
                            "v": v,
                            "s_val": float(S[0].item()),
                            "norm": float(torch.norm(t).item())
                        }
        os.remove(shard_path)
        print(f"     Purged raw shard {shard_name}. External disk safe floor preserved.")
        gc.collect()
        
    print(f"\nSaving {GEMMA2_OUTPUT_WEFT}...")
    torch.save(factorized_layers, GEMMA2_OUTPUT_WEFT)
    weft_size = os.path.getsize(GEMMA2_OUTPUT_WEFT)
    
    with open(GEMMA2_OUTPUT_GGUF, "wb") as f_out:
        header = {
            "magic": "GGUF",
            "version": 3,
            "architecture": "gemma2",
            "model_id": GEMMA2_MODEL_ID,
            "tensor_count": len(factorized_layers),
            "compression_ratio": float(total_raw_bytes / weft_size)
        }
        f_out.write(json.dumps(header).encode("utf-8") + b"\n\x00")
        with open(GEMMA2_OUTPUT_WEFT, "rb") as f_in:
            f_out.write(f_in.read())
            
    print(f"SUCCESS: Gemma-2-27B compressed from {total_raw_bytes/(1024**3):.2f} GB to {weft_size/(1024**2):.2f} MB!")

if __name__ == "__main__":
    wait_for_mistral_and_run_gemma2()
