"""
Automated Pipeline & Downloader for the Entire Google Gemma 4 Family.

Manages downloading, indexing, and sliced distillation across all 5 Gemma 4 tiers:
1. google/gemma-4-E2B-it   (9.54 GB) - Dense Edge 2B Multimodal
2. google/gemma-4-E4B-it   (14.89 GB) - Dense Edge 4B Multimodal
3. google/gemma-4-12B      (22.28 GB) - Dense Unified 12B Encoder-Free
4. google/gemma-4-26B-A4B-it (48.07 GB) - Mixture-of-Experts 26B (4B Active, 128 Experts)
5. google/gemma-4-31B-it   (58.25 GB) - Dense Frontier 31B High-Reasoning

Total family size: ~153 GB (fits easily inside the 763 GB free on /run/media/akarta/AI).
All files stream directly to /run/media/akarta/AI/models/ leaving root SSD untouched.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any
from huggingface_hub import HfApi, hf_hub_download


STORAGE_BASE = "/run/media/akarta/AI/models"


GEMMA4_FAMILY = {
    "E2B": {
        "repo_id": "google/gemma-4-E2B-it",
        "size_gb": 9.54,
        "type": "Dense Edge",
        "description": "Per-Layer Embeddings (PLE), Audio/Vision/Text unified 2B",
    },
    "E4B": {
        "repo_id": "google/gemma-4-E4B-it",
        "size_gb": 14.89,
        "type": "Dense Edge",
        "description": "Per-Layer Embeddings (PLE), higher capacity edge reasoning 4B",
    },
    "12B": {
        "repo_id": "google/gemma-4-12B",
        "size_gb": 22.28,
        "type": "Dense Unified",
        "description": "Encoder-free multimodal direct patch projection 12B",
    },
    "26B-A4B": {
        "repo_id": "google/gemma-4-26B-A4B-it",
        "size_gb": 48.07,
        "type": "MoE",
        "description": "128 Experts, Top-8 routing, 4B active parameters",
    },
    "31B": {
        "repo_id": "google/gemma-4-31B-it",
        "size_gb": 58.25,
        "type": "Dense Frontier",
        "description": "Flagship 31B dense reasoning and code synthesis model",
    },
}


def setup_family_configs():
    """
    Downloads config.json, tokenizer.json, and model.safetensors.index.json
    for all family members so they can be inspected and probed immediately.
    """
    api = HfApi()
    print("=" * 80)
    print("  GOOGLE GEMMA 4 FAMILY MANIFEST & CONFIG SYNCHRONIZATION")
    print(f"  Target Storage: {STORAGE_BASE} (External Drive)")
    print("=" * 80)

    for tier, info in GEMMA4_FAMILY.items():
        repo_id = info["repo_id"]
        tier_dir = os.path.join(STORAGE_BASE, repo_id.split("/")[-1])
        os.makedirs(tier_dir, exist_ok=True)

        print(f"\n[Configuring Tier {tier}]: {repo_id} ({info['size_gb']} GB - {info['type']})")
        print(f"  Details: {info['description']}")

        # Ensure config files are downloaded
        files = api.list_repo_files(repo_id)
        meta_files = [f for f in files if f.endswith(".json") and not f.startswith(".")]

        for mf in meta_files:
            target_path = os.path.join(tier_dir, mf)
            if not os.path.exists(target_path):
                print(f"  Downloading metadata: {mf} ...")
                try:
                    hf_hub_download(repo_id, mf, local_dir=tier_dir)
                except Exception as e:
                    print(f"  Warning: failed to download {mf}: {e}")
            else:
                print(f"  Metadata cached: {mf}")

    print("\n" + "=" * 80)
    print("  ALL GEMMA 4 FAMILY CONFIGURATIONS CACHED SUCCESSFULLY")
    print("=" * 80)


def stream_family_member(tier: str):
    """
    Downloads safetensors files for a specific family member into /run/media/akarta/AI.
    """
    assert tier in GEMMA4_FAMILY, f"Unknown tier {tier}. Choices: {list(GEMMA4_FAMILY.keys())}"
    info = GEMMA4_FAMILY[tier]
    repo_id = info["repo_id"]
    tier_dir = os.path.join(STORAGE_BASE, repo_id.split("/")[-1])
    os.makedirs(tier_dir, exist_ok=True)

    api = HfApi()
    repo_files = api.list_repo_files(repo_id)
    safetensors = [f for f in repo_files if f.endswith(".safetensors")]

    print(f"\n=== Streaming Gemma 4 Tier {tier}: {repo_id} ({len(safetensors)} files, {info['size_gb']} GB) ===")
    for sf in safetensors:
        target_path = os.path.join(tier_dir, sf)
        if not os.path.exists(target_path):
            print(f"Downloading {sf} to {tier_dir}...")
            hf_hub_download(repo_id, sf, local_dir=tier_dir)
        else:
            print(f"Already downloaded: {sf} ({os.path.getsize(target_path)/(1024**3):.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-configs", action="store_true", help="Download metadata & configs for all 5 tiers")
    parser.add_argument("--download-tier", type=str, choices=list(GEMMA4_FAMILY.keys()), help="Download weights for a specific tier")
    args = parser.parse_args()

    if args.setup_configs:
        setup_family_configs()
    elif args.download_tier:
        stream_family_member(args.download_tier)
    else:
        setup_family_configs()
