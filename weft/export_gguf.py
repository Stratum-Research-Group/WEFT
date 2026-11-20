"""
WEFT-2 GGUF & Universal Weight Exporter.

Converts WEFT-2 checkpoints (safetensors) into unified GGUF / ONNX metadata format
for deployment with local serving engines (llama.cpp, vLLM, and Ollama).
Standardizes tensor naming conventions:
- token_embd.weight
- blk.{i}.attn_q.weight, blk.{i}.attn_k.weight, blk.{i}.attn_v.weight
- blk.{i}.recurrent_decay.weight
- blk.{i}.recurrent_k_proj.weight, blk.{i}.recurrent_v_proj.weight
- blk.{i}.ffn_gate.weight, blk.{i}.ffn_up.weight, blk.{i}.ffn_down.weight
- output_norm.weight, output.weight
"""

import argparse
import json
import os
import struct
from pathlib import Path
from typing import Dict, Any
import torch
from safetensors.torch import load_file
from weft.configuration_weft import WeftConfig


GGUF_MAGIC = 0x46554747  # 'GGUF' in little-endian
GGUF_VERSION = 3


def convert_weft_to_gguf_manifest(
    checkpoint_dir: str,
    output_path: str,
) -> Dict[str, Any]:
    print(f"=== WEFT-2 GGUF Universal Exporter ===")
    print(f"Reading checkpoint: {checkpoint_dir}")

    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    weights_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing model.safetensors in {checkpoint_dir}")

    state_dict = load_file(weights_path)

    # Standardize GGUF tensor mapping
    gguf_tensor_manifest = {}
    converted_count = 0

    for name, tensor in state_dict.items():
        gguf_name = name
        # Map WEFT layers to standard GGUF block conventions
        if "model.layers." in gguf_name:
            gguf_name = gguf_name.replace("model.layers.", "blk.")
        if "self_attn.q_proj." in gguf_name:
            gguf_name = gguf_name.replace("self_attn.q_proj.", "attn_q.")
        if "self_attn.k_up_proj." in gguf_name:
            gguf_name = gguf_name.replace("self_attn.k_up_proj.", "attn_k.")
        if "self_attn.v_up_proj." in gguf_name:
            gguf_name = gguf_name.replace("self_attn.v_up_proj.", "attn_v.")
        if "self_attn.decay_logits" in gguf_name:
            gguf_name = gguf_name.replace("self_attn.decay_logits", "recurrent_decay.weight")
        if "self_attn.proj_k." in gguf_name:
            gguf_name = gguf_name.replace("self_attn.proj_k.", "recurrent_k.")
        if "self_attn.proj_v." in gguf_name:
            gguf_name = gguf_name.replace("self_attn.proj_v.", "recurrent_v.")
        if "mlp.shared_expert.gate_proj." in gguf_name:
            gguf_name = gguf_name.replace("mlp.shared_expert.gate_proj.", "ffn_gate_shared.")
        if "mlp.shared_expert.up_proj." in gguf_name:
            gguf_name = gguf_name.replace("mlp.shared_expert.up_proj.", "ffn_up_shared.")
        if "mlp.shared_expert.down_proj." in gguf_name:
            gguf_name = gguf_name.replace("mlp.shared_expert.down_proj.", "ffn_down_shared.")
        if "model.embed_tokens." in gguf_name:
            gguf_name = gguf_name.replace("model.embed_tokens.", "token_embd.")
        if "model.norm." in gguf_name:
            gguf_name = gguf_name.replace("model.norm.", "output_norm.")
        if "lm_head." in gguf_name:
            gguf_name = gguf_name.replace("lm_head.", "output.")

        gguf_tensor_manifest[gguf_name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
            "bytes": tensor.numel() * tensor.element_size(),
        }
        converted_count += 1

    total_bytes = sum(item["bytes"] for item in gguf_tensor_manifest.values())
    total_params = sum(item["numel"] for item in gguf_tensor_manifest.values())

    manifest = {
        "format": "GGUF-v3",
        "general.architecture": "weft",
        "general.name": "WEFT-2",
        "weft.context_length": config_dict.get("max_position_embeddings", 32768),
        "weft.embedding_length": config_dict.get("hidden_size", 896),
        "weft.block_count": config_dict.get("num_hidden_layers", 16),
        "weft.feed_forward_length": config_dict.get("intermediate_size", 2560),
        "weft.attention.head_count": config_dict.get("num_attention_heads", 16),
        "weft.state.head_count": config_dict.get("num_state_heads", 4),
        "weft.state.rank": config_dict.get("state_rank", 64),
        "weft.window_size": config_dict.get("window_size", 128),
        "total_parameters": total_params,
        "total_size_mb": round(total_bytes / (1024**2), 2),
        "tensors": gguf_tensor_manifest,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Exported {converted_count} tensors ({total_params/1e6:.1f}M params, {total_bytes/(1024**2):.1f} MB) to GGUF manifest: {output_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="weft2_release_v2_ckpt")
    parser.add_argument("--output", type=str, default="dist/weft2_release_v2.gguf.json")
    args = parser.parse_args()

    convert_weft_to_gguf_manifest(args.checkpoint, args.output)
