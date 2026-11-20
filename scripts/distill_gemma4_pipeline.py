"""
Live Sliced Distillation from Google Gemma 4 E2B to WEFT-2 / SULA.

Features:
- Connects directly to `/run/media/akarta/AI/models/gemma-4-E2B-it/model.safetensors`.
- Reads layer MLP weights via `safe_open` lazy slices.
- Employs SULA associative uncrossed state matching to align WEFT's linear state
  against Gemma 4's SwiGLU activation geometry.
"""

import os
import gc
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from weft import WeftConfig, WeftForCausalLM
from scripts.distill_teacher_to_weft import SlicedLayerDistiller


class Gemma4LayerProber:
    """
    Strata-style layer slicer for Gemma 4.
    Extracts activations from an arbitrary Gemma 4 layer block using zero persistent RAM.
    """

    def __init__(self, safetensors_path: str, hidden_size: int = 1536):
        self.safetensors_path = safetensors_path
        self.hidden_size = hidden_size

    def probe_layer(self, token_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
        from safetensors import safe_open

        assert os.path.exists(self.safetensors_path), f"Safetensors not found at: {self.safetensors_path}"

        with safe_open(self.safetensors_path, framework="pt", device="cpu") as f:
            # 1. Stream token embeddings for this batch
            unique_tokens, inverse_indices = torch.unique(token_ids, return_inverse=True)
            emb_key = "model.language_model.embed_tokens.weight" if "model.language_model.embed_tokens.weight" in f.keys() else "model.embed_tokens.weight"
            emb_slice = f.get_slice(emb_key)
            token_embs = torch.stack([emb_slice[int(t.item())] for t in unique_tokens])
            x = token_embs[inverse_indices].float()
            del token_embs

            # 2. Extract probed layer MLP weights
            prefix = "model.language_model.layers" if f"model.language_model.layers.{layer_idx}.mlp.gate_proj.weight" in f.keys() else "model.layers"
            gate_key = f"{prefix}.{layer_idx}.mlp.gate_proj.weight"
            up_key = f"{prefix}.{layer_idx}.mlp.up_proj.weight"
            down_key = f"{prefix}.{layer_idx}.mlp.down_proj.weight"

            w_gate = f.get_slice(gate_key)[:].float()
            w_up = f.get_slice(up_key)[:].float()
            w_down = f.get_slice(down_key)[:].float()

            # Compute SwiGLU
            act = F.silu(x @ w_gate.t()) * (x @ w_up.t())
            out = act @ w_down.t()

            del w_gate, w_up, w_down, act
            gc.collect()
            return out


def distill_gemma4_to_weft(
    gemma4_weights_path: str,
    output_dir: str = "/run/media/akarta/AI/checkpoints/gemma4_distill",
    steps: int = 10,
    batch_size: int = 2,
    seq_len: int = 32,
    layer_to_probe: int = 0,
):
    os.makedirs(output_dir, exist_ok=True)
    prober = Gemma4LayerProber(gemma4_weights_path, hidden_size=1536)

    # Configure student WEFT model
    cfg = WeftConfig(
        vocab_size=262144,  # Matching Gemma 4 tokenizer space
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=seq_len,
    )
    student = WeftForCausalLM(cfg)
    distiller = SlicedLayerDistiller(student_dim=cfg.hidden_size, teacher_dim=1536)

    optimizer = torch.optim.AdamW(list(student.parameters()) + list(distiller.parameters()), lr=1e-3)

    print(f"=== Distilling Gemma 4 Layer {layer_to_probe} into WEFT-2 (Zero-RAM Sliced) ===")
    for step in range(steps):
        tokens = torch.randint(0, 10000, (batch_size, seq_len))
        optimizer.zero_grad()

        # Student forward
        student_out = student(tokens, output_hidden_states=True)
        student_hidden = student_out.hidden_states[1]

        # Sliced teacher probe
        teacher_hidden = prober.probe_layer(tokens, layer_idx=layer_to_probe)

        loss = distiller(student_hidden, teacher_hidden)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        print(f"Step {step + 1}/{steps} | Sliced Distillation Loss: {loss.item():.4f}")

    save_path = os.path.join(output_dir, f"student_layer_{layer_to_probe}.pt")
    torch.save(student.state_dict(), save_path)
    print(f"Distilled student layer checkpoint saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", type=str, default="/run/media/akarta/AI/models/gemma-4-E2B-it/model.safetensors")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()

    if os.path.exists(args.weights_path):
        distill_gemma4_to_weft(args.weights_path, steps=args.steps, layer_to_probe=args.layer)
    else:
        print(f"Weights file not ready yet at {args.weights_path}.")
