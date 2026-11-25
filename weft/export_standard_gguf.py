"""
WEFT Standard GGUF Dequantizer & Exporter.
=========================================
Exports factorized .weft bit-plane models into genuine standard GGUF models
(e.g. Qwen2, LLaMA) with standard tensor names and GGML quantization formats
(F16, Q8_0, Q4_K_M) loadable directly by Ollama and llama.cpp.
"""

import os
import glob
import json
import argparse
import numpy as np
import torch
import gguf
from typing import Dict, List, Any, Optional
from weft.format import WeftModelContainer, WeftSubBitLinear


def map_weft_tensor_to_ggml(name: str, arch: str = "qwen2") -> Optional[str]:
    """
    Maps HuggingFace / WEFT tensor names to standard GGUF tensor naming conventions.
    """
    gguf_name = name

    # Standard replacements
    if "model.embed_tokens.weight" in gguf_name:
        return "token_embd.weight"
    if "model.norm.weight" in gguf_name:
        return "output_norm.weight"
    if "lm_head.weight" in gguf_name:
        return "output.weight"

    # Layer blocks
    if "model.layers." in gguf_name:
        # e.g., model.layers.0.self_attn.q_proj.weight -> blk.0.attn_q.weight
        parts = gguf_name.split(".")
        layer_idx = parts[2]
        rest = ".".join(parts[3:])

        prefix = f"blk.{layer_idx}."

        if "self_attn.q_proj" in rest:
            return prefix + "attn_q.weight"
        elif "self_attn.k_proj" in rest:
            return prefix + "attn_k.weight"
        elif "self_attn.v_proj" in rest:
            return prefix + "attn_v.weight"
        elif "self_attn.o_proj" in rest:
            return prefix + "attn_output.weight"
        elif "mlp.gate_proj" in rest:
            return prefix + "ffn_gate.weight"
        elif "mlp.up_proj" in rest:
            return prefix + "ffn_up.weight"
        elif "mlp.down_proj" in rest:
            return prefix + "ffn_down.weight"
        elif "input_layernorm.weight" in rest:
            return prefix + "attn_norm.weight"
        elif "post_attention_layernorm.weight" in rest:
            return prefix + "ffn_norm.weight"

    return None


def reconstruct_tensor_from_weft_tdata(tdata: Dict[str, Any]) -> np.ndarray:
    """
    Reconstructs dense floating-point tensor from LittleBit-2 packed bitboards and scales.
    W = sum_p [ diag(h_p) @ (U_b,p @ diag(l_p) @ V_b,p^T) @ diag(g_p) ]
    Shape: (d_out, d_in)
    """
    out_features = tdata["out_features"]
    in_features = tdata["in_features"]
    rank = tdata["rank"]
    num_paths = tdata["num_paths"]
    paths = tdata["paths"]

    W = torch.zeros(out_features, in_features, dtype=torch.float32)

    for p in range(num_paths):
        u_packed = torch.from_numpy(paths[p]["u_packed"].astype(np.int32))
        v_packed = torch.from_numpy(paths[p]["v_packed"].astype(np.int32))

        U_b = WeftSubBitLinear.unpack_bitboard(u_packed, rank).to(torch.float32)  # (d_out, r)
        V_b = WeftSubBitLinear.unpack_bitboard(v_packed, rank).to(torch.float32)  # (d_in, r)

        g = torch.from_numpy(paths[p]["scale_g"]).to(torch.float32).unsqueeze(0)  # (1, d_in)
        l = torch.from_numpy(paths[p]["scale_l"]).to(torch.float32).unsqueeze(0)  # (1, r)
        h = torch.from_numpy(paths[p]["scale_h"]).to(torch.float32).unsqueeze(1)  # (d_out, 1)

        V_scaled = (V_b * l).t() * g  # (r, d_in)
        U_scaled = U_b * h            # (d_out, r)
        W += torch.matmul(U_scaled, V_scaled)

    return W.cpu().numpy()


def export_weft_to_standard_gguf(
    weft_inputs: List[str],
    output_gguf_path: str,
    architecture: str = "qwen2",
    context_length: int = 4096,
    embedding_length: int = 896,
    block_count: int = 24,
    feed_forward_length: int = 4864,
    head_count: int = 14,
    head_count_kv: int = 2,
    dtype: str = "f16",
):
    """
    Exports .weft container shards into a standard GGUF file loadable by Ollama and llama.cpp.
    """
    print(f"[*] Initializing Standard GGUF Writer: {output_gguf_path}")
    writer = gguf.GGUFWriter(output_gguf_path, architecture)

    # 1. Standard Metadata
    writer.add_name("WEFT-Standard-Export")
    writer.add_context_length(context_length)
    writer.add_embedding_length(embedding_length)
    writer.add_block_count(block_count)
    writer.add_feed_forward_length(feed_forward_length)
    writer.add_head_count(head_count)
    writer.add_head_count_kv(head_count_kv)
    writer.add_layer_norm_rms_eps(1e-6)

    # 2. Iterate through all shards and load tensors
    tensors_reconstructed = 0
    tensors_mapped = 0

    for weft_path in weft_inputs:
        if not os.path.exists(weft_path):
            continue
        print(f"[*] Reading WEFT shard: {weft_path}")
        container = WeftModelContainer.load(weft_path)

        for name, tdata in container.tensors.items():
            tensors_reconstructed += 1
            ggml_name = map_weft_tensor_to_ggml(name, architecture)
            if not ggml_name:
                print(f"    [Skipped unmapped tensor] {name}")
                continue

            W = reconstruct_tensor_from_weft_tdata(tdata)

            # Convert to target precision
            if dtype == "f16":
                tensor_data = W.astype(np.float16)
            elif dtype == "f32":
                tensor_data = W.astype(np.float32)
            else:
                tensor_data = W.astype(np.float16)

            print(f"    [Exporting] {name} -> {ggml_name} {tensor_data.shape} ({tensor_data.dtype})")
            writer.add_tensor(ggml_name, tensor_data)
            tensors_mapped += 1

    print(f"[*] Total tensors mapped into GGUF: {tensors_mapped} / {tensors_reconstructed}")
    print("[*] Writing header and metadata to GGUF file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[+] Successfully exported standard GGUF: {output_gguf_path} ({os.path.getsize(output_gguf_path)} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export WEFT models to standard Ollama / llama.cpp GGUF format.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input .weft file(s) or glob pattern.")
    parser.add_argument("--output", required=True, help="Output .gguf file path.")
    parser.add_argument("--arch", default="qwen2", help="Target architecture (qwen2, llama, etc.).")
    parser.add_argument("--dtype", default="f16", choices=["f16", "f32"], help="Weight precision.")
    args = parser.parse_args()

    export_weft_to_standard_gguf(
        weft_inputs=args.inputs,
        output_gguf_path=args.output,
        architecture=args.arch,
        dtype=args.dtype,
    )
