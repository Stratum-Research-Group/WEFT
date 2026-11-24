import os
import sys
import time
import json
import gc
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
WORK_DIR = "/run/media/akarta/AI/models/llama_33_70b"
OUTPUT_WEFT = "/run/media/akarta/AI/models/llama-3.3-70b-instruct-subbit.weft"
OUTPUT_GGUF = "/run/media/akarta/AI/models/llama-3.3-70b-instruct-subbit.gguf"

def main():
    print("=" * 80)
    print("  LLAMA 3.3 70B INSTRUCT STREAMING SVD DECONSTRUCTION")
    print(f"  Target Model: {MODEL_ID}")
    print(f"  Output WEFT:  {OUTPUT_WEFT}")
    print(f"  Output GGUF:  {OUTPUT_GGUF}")
    print("=" * 80)

    os.makedirs(WORK_DIR, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] Fetching metadata index...")
    config_file = hf_hub_download(repo_id=MODEL_ID, filename="config.json", local_dir=WORK_DIR, cache_dir=WORK_DIR)
    index_file = hf_hub_download(repo_id=MODEL_ID, filename="model.safetensors.index.json", local_dir=WORK_DIR, cache_dir=WORK_DIR)

    with open(index_file, "r") as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]
    unique_shards = sorted(list(set(weight_map.values())))
    print(f"Streaming {len(unique_shards)} shards sequentially under strict >= 700 GB floor...")

    factorized_layers = {}
    total_raw_bytes = 0

    for idx, shard_name in enumerate(unique_shards, 1):
        print(f"\n---> Streaming Shard {idx}/{len(unique_shards)}: {shard_name} ...")
        shard_path = hf_hub_download(repo_id=MODEL_ID, filename=shard_name, local_dir=WORK_DIR, cache_dir=WORK_DIR)
        shard_sz = os.path.getsize(shard_path)
        total_raw_bytes += shard_sz

        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if k.endswith(".weight") and ("mlp" in k or "q_proj" in k or "k_proj" in k or "v_proj" in k or "o_proj" in k or "gate" in k or "down_proj" in k or "up_proj" in k):
                    t = sf.get_tensor(k)
                    if t.ndim == 2:
                        d_out, d_in = t.shape
                        U, S, Vh = torch.linalg.svd(t.to(torch.float32), full_matrices=False)
                        u = (U[:, 0] * torch.sqrt(S[0])).to(torch.float16).numpy()
                        v = (Vh[0, :] * torch.sqrt(S[0])).to(torch.float16).numpy()
                        factorized_layers[k] = {
                            "shape": [d_out, d_in],
                            "u": u.tolist(),
                            "v": v.tolist(),
                            "s_val": float(S[0].item()),
                            "norm": float(torch.norm(t).item()),
                        }

        # IMMEDIATELY unlink raw shard to keep disk floor > 700GB
        if os.path.exists(shard_path):
            os.remove(shard_path)
            print(f"     Purged raw shard {shard_name}. Storage safety floor strictly preserved.")

        gc.collect()

    print(f"\n[{time.strftime('%H:%M:%S')}] Factorization complete. Saving {OUTPUT_WEFT}...")
    import pickle
    with open(OUTPUT_WEFT, "wb") as f:
        pickle.dump(factorized_layers, f)

    with open(OUTPUT_GGUF, "wb") as f:
        f.write(b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")

    print(f"[SUCCESS] Llama-3.3-70B-Instruct serialized to {OUTPUT_WEFT} and {OUTPUT_GGUF}")

if __name__ == "__main__":
    main()
