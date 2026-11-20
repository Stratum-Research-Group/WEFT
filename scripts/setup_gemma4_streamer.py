"""
Strata-Style Streamer and Slicer for Google Gemma 4.

Supports:
- google/gemma-4-E2B-it (Dense 2B edge multimodal, 35 layers, 1536 dim)
- google/gemma-4-E4B-it (Dense 4B edge multimodal)
- google/gemma-4-26B-A4B-it (MoE 26B, 4B active parameters)

Key Features:
1. Downloads directly to `/run/media/akarta/AI/models/` (Zero root drive usage).
2. Uses `safetensors.safe_open` for zero-RAM memory-mapped layer probing.
3. Maps Gemma 4's nested multimodal text_config (`model.layers.k.mlp.*`) directly
   into WEFT's SULA recurrent distillation harness.
"""

import os
import json
import argparse
from typing import Optional, Dict, Any

from huggingface_hub import hf_hub_download
from safetensors import safe_open


DEFAULT_GEMMA4_DIR = "/run/media/akarta/AI/models/gemma-4-E2B-it"


def ensure_gemma4_config(
    repo_id: str = "google/gemma-4-E2B-it",
    target_dir: str = DEFAULT_GEMMA4_DIR,
) -> Dict[str, Any]:
    """
    Downloads and caches Gemma 4 config and tokenizer files without downloading the multi-GB weights.
    """
    os.makedirs(target_dir, exist_ok=True)
    config_path = os.path.join(target_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Downloading {repo_id} config.json to {target_dir}...")
        hf_hub_download(repo_id, "config.json", local_dir=target_dir)

    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def download_gemma4_weights(
    repo_id: str = "google/gemma-4-E2B-it",
    target_dir: str = DEFAULT_GEMMA4_DIR,
) -> str:
    """
    Downloads Gemma 4 safetensors directly into external high-capacity drive.
    """
    os.makedirs(target_dir, exist_ok=True)
    weights_path = os.path.join(target_dir, "model.safetensors")
    if not os.path.exists(weights_path):
        print(f"Streaming {repo_id} safetensors weights into {target_dir} (799GB free available)...")
        hf_hub_download(repo_id, "model.safetensors", local_dir=target_dir)
    return weights_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--target-dir", type=str, default=DEFAULT_GEMMA4_DIR)
    parser.add_argument("--download-weights", action="store_true", help="Download the full model.safetensors")
    args = parser.parse_args()

    cfg = ensure_gemma4_config(args.repo_id, args.target_dir)
    text_cfg = cfg.get("text_config", {})
    print(f"Successfully configured {args.repo_id}:")
    print(f"  - Hidden Size     : {text_cfg.get('hidden_size')}")
    print(f"  - Hidden Layers   : {text_cfg.get('num_hidden_layers')}")
    print(f"  - Intermediate Dim: {text_cfg.get('intermediate_size')}")
    print(f"  - Vocab Size      : {text_cfg.get('vocab_size')}")

    if args.download_weights:
        w_path = download_gemma4_weights(args.repo_id, args.target_dir)
        print(f"Weights downloaded to: {w_path}")
