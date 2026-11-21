"""
Streaming Sharded Quantization Engine for Extreme 70B+ Foundation Models.
Downloads model shards one-by-one, decomposes linear weights into sub-bit .weft bitboards,
and deletes each raw uncompressed shard immediately to bound peak disk consumption.
"""

import os
import gc
import json
import time
import argparse
import torch
from huggingface_hub import hf_hub_download, HfApi
from safetensors import safe_open

from weft.converter import factorize_linear_layer
from weft.format import WeftModelContainer


def stream_quantize_hf_repo(
    repo_id: str,
    output_dir: str,
    base_rank: int = 64,
    itq_iterations: int = 3,
    max_shards: int = None
):
    """
    Streams shards from Hugging Face, quantizes them to sub-bit .weft,
    and removes raw safetensors to preserve disk capacity.
    """
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    
    # Identify safetensors shards
    shards = sorted([s.rfilename for s in info.siblings if s.rfilename.endswith('.safetensors') and 'model-' in s.rfilename])
    if not shards:
        # Check single safetensors
        shards = [s.rfilename for s in info.siblings if s.rfilename.endswith('.safetensors')]
        
    print(f"============================================================")
    print(f" Streaming Quantization Pipeline: {repo_id}")
    print(f" Total Shards: {len(shards)} | Target Output: {output_dir}")
    print(f"============================================================")
    
    os.makedirs(output_dir, exist_ok=True)
    temp_download_dir = os.path.join(output_dir, ".stream_staging")
    os.makedirs(temp_download_dir, exist_ok=True)
    
    if max_shards:
        shards = shards[:max_shards]
        print(f"Limiting execution to first {max_shards} shards.")

    total_t0 = time.time()
    
    for idx, shard_name in enumerate(shards):
        weft_shard_path = os.path.join(output_dir, f"{os.path.splitext(shard_name)[0]}_subbit.weft")
        if os.path.exists(weft_shard_path):
            print(f"\n>>> [Shard {idx+1}/{len(shards)}] Skipping {shard_name} (Already quantized: {weft_shard_path})")
            continue

        print(f"\n>>> [Shard {idx+1}/{len(shards)}] Downloading {shard_name}...")
        t_dl = time.time()
        shard_path = hf_hub_download(
            repo_id=repo_id,
            filename=shard_name,
            local_dir=temp_download_dir,
            local_dir_use_symlinks=False
        )
        dl_size_mb = os.path.getsize(shard_path) / 1e6
        print(f"Downloaded {shard_name} ({dl_size_mb:.2f} MB in {time.time() - t_dl:.1f}s)")
        
        container = WeftModelContainer(metadata={
            "repo_id": repo_id,
            "shard_name": shard_name,
            "base_rank": base_rank,
            "quant_engine": "weft-joint-itq"
        })
        
        t_q = time.time()
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            linear_keys = [k for k in keys if "weight" in k and len(f.get_slice(k).get_shape()) == 2]
            print(f"Factorizing {len(linear_keys)} linear matrices in {shard_name}...")
            
            for k_idx, k in enumerate(linear_keys):
                W = f.get_tensor(k).float()
                layer = factorize_linear_layer(
                    W,
                    base_rank=base_rank,
                    num_paths=1,
                    itq_iterations=itq_iterations
                )
                container.add_subbit_linear(k, layer)
                if (k_idx + 1) % 5 == 0 or (k_idx + 1) == len(linear_keys):
                    print(f"  [{k_idx+1}/{len(linear_keys)}] Processed {k:<55} -> {layer.bits_per_weight():.4f} BPW")
                    
        container.save(weft_shard_path)
        weft_size_mb = os.path.getsize(weft_shard_path) / 1e6
        ratio = dl_size_mb / max(0.01, weft_size_mb)
        print(f"Quantized shard saved: {weft_shard_path} ({weft_size_mb:.2f} MB | {ratio:.1f}x compression)")
        
        # Immediate cleanup of raw heavy file to strictly preserve disk capacity
        if os.path.exists(shard_path):
            os.remove(shard_path)
            print(f"Removed temporary raw shard {shard_path} (reclaimed {dl_size_mb:.2f} MB)")
            
        # Auto-update consolidated GGUF container
        try:
            from weft.export_subbit_gguf import export_multiple_weft_to_gguf
            all_shards = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith("_subbit.weft")])
            if all_shards:
                gguf_target = os.path.join(output_dir, "model_consolidated_subbit.gguf")
                export_multiple_weft_to_gguf(all_shards, gguf_target)
        except Exception as e_gguf:
            print(f"Auto-GGUF consolidation skipped: {e_gguf}")

        gc.collect()

    # Clean temporary staging directory
    try:
        os.rmdir(temp_download_dir)
    except Exception:
        pass
        
    print(f"\n============================================================")
    print(f"Streaming Quantization Complete for {repo_id} in {time.time() - total_t0:.1f}s!")
    print(f"============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True, help="HF model repository ID")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save .weft shards")
    parser.add_argument("--base-rank", type=int, default=64, help="Base rank for factorizations")
    parser.add_argument("--itq-iterations", type=int, default=3, help="ITQ polar iterations")
    parser.add_argument("--max-shards", type=int, default=None, help="Limit number of shards to process")
    args = parser.parse_args()
    
    stream_quantize_hf_repo(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        base_rank=args.base_rank,
        itq_iterations=args.itq_iterations,
        max_shards=args.max_shards
    )
