"""
BitNet b1.58 Ternary Quantization Pipeline for Distilled WEFT Weights.

Quantizes distilled FP32 weights to ternary {-1, 0, +1} via absolute mean scaling,
enabling pure integer addition GEMM forward passes with zero multiplication.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from weft import WeftConfig, WeftForCausalLM


def quantize_weight_ternary(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes full-precision weights to ternary {-1, 0, +1} using BitNet b1.58 formulation.
    Returns:
        - ternary_w: Int8 tensor with values in {-1, 0, +1}
        - scale: FP32 scale factor = mean(abs(w))
    """
    scale = w.abs().mean().clamp_min(1e-5)
    scaled_w = w / scale
    ternary_w = torch.round(scaled_w).clamp(-1, 1).to(torch.int8)
    return ternary_w, scale


def quantize_distilled_model_to_bitnet(
    ckpt_path: str = "/run/media/akarta/AI/checkpoints/deep_distill/student_deep_distilled_weft.pt",
    output_path: str = "/run/media/akarta/AI/checkpoints/deep_distill/student_ternary_bitweft.pt",
) -> Dict[str, float]:
    print("=" * 80)
    print("  BITNET b1.58 TERNARY QUANTIZATION PIPELINE")
    print(f"  Input Checkpoint: {ckpt_path}")
    print("=" * 80)

    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    state_dict = torch.load(ckpt_path, weights_only=True)

    ternary_dict = {}
    scales_dict = {}
    total_weights = 0
    zero_weights = 0

    for name, param in state_dict.items():
        if "weight" in name and param.dim() >= 2:
            t_w, scale = quantize_weight_ternary(param)
            ternary_dict[name] = t_w
            scales_dict[name + "_scale"] = scale
            total_weights += t_w.numel()
            zero_weights += (t_w == 0).sum().item()
        else:
            ternary_dict[name] = param

    sparsity = (zero_weights / max(1, total_weights)) * 100.0
    torch.save({"weights": ternary_dict, "scales": scales_dict}, output_path)

    # Compute uncompressed vs compressed size
    uncompressed_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    compressed_mb = os.path.getsize(output_path) / (1024 * 1024)
    compression_ratio = uncompressed_mb / max(0.01, compressed_mb)

    print(f"  Quantization Complete:")
    print(f"  - Total Quantized Weights: {total_weights:,}")
    print(f"  - Zero Weights (Sparsity) : {sparsity:.2f}%")
    print(f"  - Uncompressed Size (FP32): {uncompressed_mb:.2f} MB")
    print(f"  - Ternary Pack Size (INT8): {compressed_mb:.2f} MB")
    print(f"  - Memory Reduction Factor : {compression_ratio:.2f}x")
    print(f"  Saved ternary model to    : {output_path}")
    print("=" * 80)

    return {
        "sparsity": sparsity,
        "compression_ratio": compression_ratio,
        "uncompressed_mb": uncompressed_mb,
        "compressed_mb": compressed_mb,
    }


if __name__ == "__main__":
    quantize_distilled_model_to_bitnet()
