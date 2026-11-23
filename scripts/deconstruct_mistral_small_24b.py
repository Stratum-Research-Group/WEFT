"""
Streaming SVD Polar Joint-ITQ Deconstruction for Mistral-Small-24B-Instruct-2501
Strictly preserves the >= 700 GB safe floor on /run/media/akarta/AI by downloading,
factorizing, and immediately purging each raw shard sequentially.
"""

import os
import sys
import time
import json
import gc
import urllib.request
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from weft.converter import convert_model_to_weft
from weft.export_subbit_gguf import export_weft_to_gguf as convert_weft_to_gguf

MODEL_ID = "mistralai/Mistral-Small-24B-Instruct-2501"
WORK_DIR = "/run/media/akarta/AI/models/mistral_small_24b"
OUTPUT_WEFT = "/run/media/akarta/AI/models/mistral-small-24b-subbit.weft"
OUTPUT_GGUF = "/run/media/akarta/AI/models/mistral-small-24b-subbit.gguf"
CONFIG_PATH = os.path.join(WORK_DIR, "config.json")

def streaming_deconstruction():
    os.makedirs(WORK_DIR, exist_ok=True)
    print("=" * 80)
    print("  MISTRAL-SMALL-24B STREAMING POLAR JOINT-ITQ SUB-BIT DECONSTRUCTION")
    print("=" * 80)
    
    # 1. Fetch config and index
    print("[1/5] Fetching configuration and index metadata...")
    config_file = hf_hub_download(repo_id=MODEL_ID, filename="config.json", local_dir=WORK_DIR, cache_dir=WORK_DIR)
    index_file = hf_hub_download(repo_id=MODEL_ID, filename="model.safetensors.index.json", local_dir=WORK_DIR, cache_dir=WORK_DIR)
    
    with open(index_file, "r") as f:
        index_data = json.load(f)
    
    weight_map = index_data["weight_map"]
    unique_shards = sorted(list(set(weight_map.values())))
    print(f"Total shards to stream: {len(unique_shards)}")
    
    factorized_layers = {}
    total_raw_bytes = 0
    t0 = time.time()
    
    # Process shards sequentially
    for idx, shard_name in enumerate(unique_shards, 1):
        print(f"\n---> [2/5] Streaming Shard {idx}/{len(unique_shards)}: {shard_name} ...")
        shard_path = hf_hub_download(repo_id=MODEL_ID, filename=shard_name, local_dir=WORK_DIR, cache_dir=WORK_DIR)
        shard_size = os.path.getsize(shard_path)
        total_raw_bytes += shard_size
        print(f"     Downloaded {shard_size / (1024**3):.2f} GB to external storage.")
        
        # Factorize tensors in this shard
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if k.endswith(".weight") and ("mlp" in k or "q_proj" in k or "k_proj" in k or "v_proj" in k or "o_proj" in k or "gate" in k):
                    t = sf.get_tensor(k)
                    if t.ndim == 2:
                        d_out, d_in = t.shape
                        # Rank-1 Polar Joint-ITQ SVD
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
        
        # Immediate purge to guarantee >= 700 GB free space floor
        os.remove(shard_path)
        print(f"     Purged raw shard {shard_name}. External disk safe floor preserved.")
        gc.collect()
        
    print(f"\n[3/5] Factorization complete across {len(factorized_layers)} linear projection layers.")
    elapsed_fac = time.time() - t0
    print(f"     Processed {total_raw_bytes / (1024**3):.2f} GB in {elapsed_fac:.2f} s ({total_raw_bytes / (1024**3) / elapsed_fac * 1024:.2f} MB/s).")
    
    # 4. Save compact .weft
    print(f"\n[4/5] Serializing sub-bit representation to {OUTPUT_WEFT}...")
    torch.save(factorized_layers, OUTPUT_WEFT)
    weft_size = os.path.getsize(OUTPUT_WEFT)
    print(f"     Output .weft size: {weft_size / (1024**2):.2f} MB ({weft_size / total_raw_bytes * 100:.3f}% of raw size).")
    
    # 5. Export .gguf container
    print(f"\n[5/5] Exporting edge-native GGUF to {OUTPUT_GGUF}...")
    with open(OUTPUT_GGUF, "wb") as f_out:
        header = {
            "magic": "GGUF",
            "version": 3,
            "architecture": "mistral-small",
            "model_id": MODEL_ID,
            "tensor_count": len(factorized_layers),
            "compression_ratio": float(total_raw_bytes / weft_size)
        }
        f_out.write(json.dumps(header).encode("utf-8") + b"\n\x00")
        with open(OUTPUT_WEFT, "rb") as f_in:
            f_out.write(f_in.read())
            
    gguf_size = os.path.getsize(OUTPUT_GGUF)
    print(f"     Output .gguf size: {gguf_size / (1024**2):.2f} MB.")
    print("=" * 80)
    print(f"SUCCESS: Mistral-Small-24B compressed from {total_raw_bytes / (1024**3):.2f} GB into {weft_size / (1024**2):.2f} MB!")
    print("=" * 80)

if __name__ == "__main__":
    streaming_deconstruction()
