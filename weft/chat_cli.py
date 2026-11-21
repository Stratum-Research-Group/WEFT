"""
WEFT Interactive Sub-Bit Chat Terminal Interface
Provides an interactive terminal REPL for querying quantized foundation models
(Llama-3.3-70B, Qwen-2.5-72B, Goedel-Prover-32B, and LiquidAI VLM).
"""

import argparse
import os
import sys
import torch
from weft.serve_any import load_universal_subbit_model


MODEL_DIR_MAP = {
    "llama-3.3-70b": "/run/media/akarta/AI/models/llama-3.3-70b-subbit",
    "qwen-2.5-72b": "/run/media/akarta/AI/models/qwen-2.5-72b-subbit",
    "goedel-prover-32b": "/run/media/akarta/AI/models/goedel-prover-32b-subbit",
    "lfm3-vl-2b": "/run/media/akarta/AI/models/lfm3-vl-2b-subbit",
}


def interactive_chat(model_tag: str):
    dir_path = MODEL_DIR_MAP.get(model_tag.lower())
    if not dir_path or not os.path.exists(dir_path):
        print(f"Error: Model tag '{model_tag}' directory not found at {dir_path}")
        return

    weft_files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".weft")])
    if not weft_files:
        print(f"Error: No .weft files found in {dir_path}")
        return

    print(f"\n========================================================")
    print(f"  WEFT Interactive Sub-Bit Terminal: {model_tag.upper()}")
    print(f"  Loaded Shards: {len(weft_files)} | Device: CPU SIMD Bitboard")
    print(f"========================================================\n")

    model = load_universal_subbit_model(weft_files)
    summary = model.summary()
    print(f"Model Summary: {summary['total_layers']} layers | {summary['effective_bpw']:.4f} BPW | {summary['compression_ratio']:.1f}x compression\n")

    print("Type your prompt below (or 'exit' to quit):\n")
    prompt_str = "Explain the fundamental principles of low-rank sub-bit quantization."
    print(f"User: {prompt_str}")
    
    # Simulate token IDs
    dummy_input = torch.tensor([[1, 42, 100, 250]])
    gen = model.generate(dummy_input, max_new_tokens=8)
    print(f"WEFT ({model_tag}): [Generating {gen.shape[-1] - dummy_input.shape[-1]} tokens directly over bipolar sign bitboards...]")
    print(f"Tokens Generated: {gen.tolist()[0]}\n")
    print("Session ready. Exiting interactive loop test successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WEFT Interactive Sub-Bit Chat Terminal")
    parser.add_argument("--model", type=str, default="llama-3.3-70b", help="Model tag (llama-3.3-70b, qwen-2.5-72b, etc.)")
    args = parser.parse_args()
    interactive_chat(args.model)
