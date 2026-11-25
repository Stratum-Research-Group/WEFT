import os
import sys
import json
import time
import subprocess
import datetime
import torch
import numpy as np
from weft.configuration_weft import WeftConfig

def get_environment_info(device_name="cpu"):
    commit = "unknown"
    dirty = False
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        pass

    gpu_info = {}
    if torch.cuda.is_available() and device_name != "cpu":
        props = torch.cuda.get_device_properties(0)
        gpu_info = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": props.total_memory / (1024**3),
            "device_count": torch.cuda.device_count()
        }

    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": commit,
        "git_commit_short": commit[:7],
        "git_dirty": dirty,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_info,
        "device": device_name
    }

def get_tiny_config(**kwargs):
    default_kwargs = dict(
        vocab_size=257,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_state_heads=2,
        state_rank=8,
        window_size=8,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
    )
    default_kwargs.update(kwargs)
    return WeftConfig(**default_kwargs)

def set_seed(seed=1234):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
