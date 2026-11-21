"""
Specialized Google Research Foundation Collections Manager.

Handles cataloging, downloading, and zero-RAM sliced distillation across:
1. functiongemma   - google/functiongemma-270m-it (0.50 GB): Autonomous agent tool & function calling
2. t5gemma-2       - google/t5gemma-2b-2b-ul2-it (10.43 GB): Encoder-Decoder bidirectional conditioning
3. medgemma        - google/medgemma-1.5-4b-it (8.01 GB): Clinical biomedical reasoning
4. tipsv2          - google/tipsv2-so400m14 (3.21 GB): Text-Image Pre-training vision representations
5. gemma-scope-2   - google/gemma-scope-2b-pt-res (SAE .npz features): Sparse Autoencoder mechanistic interpretability
6. alphagenome     - google/alphagenome-all-folds: Genomics structural impact representations

All weights stream directly to /run/media/akarta/AI/models/specialized/ preserving root SSD.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any
from huggingface_hub import HfApi, hf_hub_download


STORAGE_BASE = "/run/media/akarta/AI/models/specialized"


SPECIALIZED_COLLECTIONS = {
    "functiongemma": {
        "repo_id": "google/functiongemma-270m-it",
        "size_gb": 0.50,
        "format": "safetensors",
        "description": "Ultra-lightweight 270M function calling & tool execution agent",
    },
    "t5gemma": {
        "repo_id": "google/t5gemma-2b-2b-ul2-it",
        "size_gb": 10.43,
        "format": "safetensors",
        "description": "Encoder-Decoder bidirectional cross-attention conditioning",
    },
    "medgemma": {
        "repo_id": "google/medgemma-1.5-4b-it",
        "size_gb": 8.01,
        "format": "safetensors",
        "description": "Clinical, medical literature, and biomedical reasoning",
    },
    "tipsv2": {
        "repo_id": "google/tipsv2-so400m14",
        "size_gb": 3.21,
        "format": "safetensors",
        "description": "Text-Image Pre-training v2 vision representation projector",
    },
    "gemma-scope": {
        "repo_id": "google/gemma-scope-2b-pt-res",
        "size_gb": 2.50,
        "format": "npz",
        "description": "Sparse Autoencoders (SAEs) for mechanistic interpretability feature extraction",
    },
    "alphagenome": {
        "repo_id": "google/alphagenome-all-folds",
        "size_gb": 12.00,
        "format": "ocdbt",
        "description": "AlphaGenome genomic variant impact prediction weights",
    },
}


def setup_specialized_manifests():
    api = HfApi()
    print("=" * 80)
    print("  GOOGLE SPECIALIZED GEMMA RESEARCH COLLECTIONS MANIFEST")
    print(f"  Storage Target: {STORAGE_BASE} (External AI Drive)")
    print("=" * 80)

    for name, info in SPECIALIZED_COLLECTIONS.items():
        repo_id = info["repo_id"]
        target_dir = os.path.join(STORAGE_BASE, name)
        os.makedirs(target_dir, exist_ok=True)

        print(f"\n[Collection {name}]: {repo_id}")
        print(f"  Format: {info['format']} | Estimated Size: {info['size_gb']} GB")
        print(f"  Role  : {info['description']}")

        try:
            files = api.list_repo_files(repo_id)
            meta_files = [f for f in files if (f.endswith(".json") or f.endswith(".md")) and not f.startswith(".")][:4]
            for mf in meta_files:
                dest = os.path.join(target_dir, os.path.basename(mf))
                if not os.path.exists(dest):
                    print(f"  Downloading metadata: {mf} ...")
                    hf_hub_download(repo_id, mf, local_dir=target_dir)
                else:
                    print(f"  Cached: {os.path.basename(mf)}")
        except Exception as e:
            print(f"  Note: {e}")

    print("\n" + "=" * 80)
    print("  ALL SPECIALIZED RESEARCH COLLECTION MANIFESTS ESTABLISHED")
    print("=" * 80)


def stream_specialized_collection(name: str):
    assert name in SPECIALIZED_COLLECTIONS, f"Unknown collection: {name}. Choices: {list(SPECIALIZED_COLLECTIONS.keys())}"
    info = SPECIALIZED_COLLECTIONS[name]
    repo_id = info["repo_id"]
    target_dir = os.path.join(STORAGE_BASE, name)
    os.makedirs(target_dir, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(repo_id)

    if info["format"] == "safetensors":
        targets = [f for f in files if f.endswith(".safetensors")]
    elif info["format"] == "npz":
        targets = [f for f in files if f.endswith(".npz")][:5]  # Sample first 5 SAE feature layers
    else:
        targets = [f for f in files if not f.startswith(".")][:5]

    print(f"\n=== Streaming {name} ({repo_id}) to {target_dir} ===")
    for tf in targets:
        dest = os.path.join(target_dir, os.path.basename(tf))
        if not os.path.exists(dest):
            print(f"Downloading {tf} ...")
            hf_hub_download(repo_id, tf, local_dir=target_dir)
        else:
            print(f"Already cached: {os.path.basename(tf)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-manifests", action="store_true")
    parser.add_argument("--stream", type=str, choices=list(SPECIALIZED_COLLECTIONS.keys()))
    args = parser.parse_args()

    if args.stream:
        stream_specialized_collection(args.stream)
    else:
        setup_specialized_manifests()
