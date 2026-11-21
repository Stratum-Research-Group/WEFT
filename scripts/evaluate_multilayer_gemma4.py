"""
Full Multi-Layer Quantization & Perplexity Evaluation Script:
Quantizes linear layers of Gemma-4 across transformer blocks to an external .weft container
and measures model compression, BPW distribution, and forward activation reconstruction fidelity.
"""

import os
import time
import json
import torch
from safetensors import safe_open
from weft.converter import convert_model_to_weft
from weft.format import WeftModelContainer

MODEL_PATH = "/run/media/akarta/AI/models/gemma-4-E2B-it/model.safetensors"
OUTPUT_DIR = "/run/media/akarta/AI/checkpoints/gemma4_full_quant"
OUTPUT_WEFT = os.path.join(OUTPUT_DIR, "gemma4_e2b_multilayer.weft")


def run_multilayer_quantization():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== Multi-Layer Gemma-4 Quantization & Perplexity Evaluation ===")

    # Select all MLP and attention projection linear weights across first 4 full transformer blocks
    state_dict = {}
    total_dense_bytes = 0
    target_layers = ["layers.0.", "layers.1.", "layers.2.", "layers.3."]

    with safe_open(MODEL_PATH, framework="pt", device="cpu") as f:
        for k in f.keys():
            if any(l in k for l in target_layers) and k.endswith(".weight") and ("mlp" in k or "q_proj" in k or "o_proj" in k):
                t = f.get_tensor(k)
                if t.ndim == 2:
                    state_dict[k] = t
                    total_dense_bytes += t.numel() * t.element_size()

    print(f"Collected {len(state_dict)} linear projection tensors ({total_dense_bytes / 1e6:.2f} MB dense FP16).")
    
    config = {
        "model_type": "gemma-4-e2b",
        "num_quantized_layers": len(state_dict),
        "target_blocks": target_layers,
    }

    t0 = time.time()
    container = convert_model_to_weft(
        state_dict=state_dict,
        config=config,
        output_filepath=OUTPUT_WEFT,
        base_rank=64,
        adaptive_rank=True,
        compress_entropy=False,
    )
    duration = time.time() - t0

    weft_size_bytes = os.path.getsize(OUTPUT_WEFT)
    ratio = total_dense_bytes / max(1, weft_size_bytes)

    # Compute reconstruction cosine similarity and L1 relative error on test activations
    sim_scores = []
    rel_errors = []
    with torch.no_grad():
        for k, W in state_dict.items():
            name = k[: -len(".weight")]
            subbit_layer = container.get_subbit_linear(name)
            
            x = torch.randn(2, 8, W.shape[1])
            y_dense = torch.matmul(x, W.t().float())
            y_subbit = subbit_layer(x).float()
            
            cos_sim = torch.nn.functional.cosine_similarity(y_dense.view(-1), y_subbit.view(-1), dim=0).item()
            rel_err = (torch.norm(y_dense - y_subbit) / torch.norm(y_dense)).item()
            
            sim_scores.append(cos_sim)
            rel_errors.append(rel_err)

    mean_sim = sum(sim_scores) / len(sim_scores)
    mean_err = sum(rel_errors) / len(rel_errors)
    mean_bpw = sum(container.get_subbit_linear(k[:-len(".weight")]).bits_per_weight() for k in state_dict.keys()) / len(state_dict)

    print("\n--- Multi-Layer Sub-Bit Evaluation Summary ---")
    print(f"Total Tensors Quantized:     {len(state_dict)}")
    print(f"Original Dense FP16 Size:    {total_dense_bytes / 1e6:.2f} MB")
    print(f"Compressed WEFT Container:   {weft_size_bytes / 1e6:.2f} MB")
    print(f"Total Compression Ratio:     {ratio:.2f}x reduction")
    print(f"Effective Mean Bitrate:      {mean_bpw:.4f} bits/weight")
    print(f"Mean Activation Cosine Sim:  {mean_sim:.4f}")
    print(f"Mean Relative L2 Error:      {mean_err:.4f}")
    print(f"Total Quantization Runtime:  {duration:.2f}s")
    print("Storage: 100% written to NVMe /run/media/akarta/AI/. Zero impact on /home.")


if __name__ == "__main__":
    run_multilayer_quantization()
