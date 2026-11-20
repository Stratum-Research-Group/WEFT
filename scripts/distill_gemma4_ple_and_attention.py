"""
Distillation of Gemma 4 Per-Layer Embeddings (PLE) and Attention Kernels into WEFT SULA.

Distills:
1. Per-Layer Embeddings (PLE):
   Gemma 4's `embed_tokens_per_layer` [262144, 8960] + `per_layer_projection` [1536, 256]
   into WEFT's SULA associative rank-r projection `r_proj`.
2. QK-Normalized Attention Associative Geometry:
   Aligns WEFT's recurrent linear attention state readout against Gemma 4's multi-head
   attention projection (q_proj, k_proj, v_proj, o_proj) with QK-norm scaling.
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

from weft import WeftConfig
from weft.singlet_attention import SingletUncrossedAttention


class Gemma4AttentionPLEProber:
    """
    Strata-style zero-RAM prober for Gemma 4 Per-Layer Embeddings and Attention blocks.
    """

    def __init__(self, safetensors_path: str = "/run/media/akarta/AI/models/gemma-4-E2B-it/model.safetensors"):
        self.safetensors_path = safetensors_path

    def probe_ple_and_attention(
        self,
        token_ids: torch.Tensor,
        layer_idx: int = 0,
    ) -> dict:
        """
        Streams only the PLE slice and self_attn weights for layer_idx.
        Returns:
            - ple_feature: [B, S, 256] low-rank layer token representation
            - attn_out: [B, S, 1536] attention block output
        """
        from safetensors import safe_open

        with safe_open(self.safetensors_path, framework="pt", device="cpu") as f:
            b, s = token_ids.shape
            unique_tokens, inverse_indices = torch.unique(token_ids, return_inverse=True)

            # 1. Base embedding
            emb_slice = f.get_slice("model.language_model.embed_tokens.weight")
            token_embs = torch.stack([emb_slice[int(t.item())] for t in unique_tokens])
            x = token_embs[inverse_indices].float()  # [B, S, 1536]
            del token_embs

            # 2. Per-Layer Embedding (PLE) slice
            # Each layer uses a 256-dim slice of the 8960-dim per-layer embedding table (35 layers * 256 = 8960)
            ple_slice = f.get_slice("model.language_model.embed_tokens_per_layer.weight")
            # Slice the 256 columns corresponding to this specific layer
            layer_offset = layer_idx * 256
            ple_token_embs = torch.stack([ple_slice[int(t.item()), layer_offset : layer_offset + 256] for t in unique_tokens])
            ple_features = ple_token_embs[inverse_indices].float()  # [B, S, 256]
            del ple_token_embs

            # 3. Attention projections
            prefix = f"model.language_model.layers.{layer_idx}.self_attn"
            w_q = f.get_slice(f"{prefix}.q_proj.weight")[:].float()      # [2048, 1536]
            w_k = f.get_slice(f"{prefix}.k_proj.weight")[:].float()      # [256, 1536]
            w_v = f.get_slice(f"{prefix}.v_proj.weight")[:].float()      # [256, 1536]
            w_o = f.get_slice(f"{prefix}.o_proj.weight")[:].float()      # [1536, 2048]

            q_norm = f.get_slice(f"{prefix}.q_norm.weight")[:].float()   # [256]
            k_norm = f.get_slice(f"{prefix}.k_norm.weight")[:].float()   # [256]

            # Forward Gemma 4 QK-Norm Attention
            q = x @ w_q.t()  # [B, S, 2048]
            k = x @ w_k.t()  # [B, S, 256]
            v = x @ w_v.t()  # [B, S, 256]

            # Head reshape: Gemma 4 E2B has 8 Q heads of 256 dim, 1 KV head (MQA) of 256 dim
            num_q_heads = 8
            head_dim = 256
            q = q.view(b, s, num_q_heads, head_dim).transpose(1, 2)  # [B, 8, S, 256]
            k = k.view(b, s, 1, head_dim).transpose(1, 2)            # [B, 1, S, 256]
            v = v.view(b, s, 1, head_dim).transpose(1, 2)            # [B, 1, S, 256]

            # QK RMSNorm scaling
            q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6) * q_norm.view(1, 1, 1, head_dim)
            k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6) * k_norm.view(1, 1, 1, head_dim)

            scale = 1.0 / (head_dim ** 0.5)
            scores = (q @ k.transpose(-1, -2)) * scale  # [B, 8, S, S]
            causal_mask = torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
            attn_weights = torch.softmax(scores + causal_mask, dim=-1)
            attn_out = attn_weights @ v  # [B, 8, S, 256]

            attn_flat = attn_out.transpose(1, 2).contiguous().view(b, s, 2048)
            o_out = attn_flat @ w_o.t()  # [B, S, 1536]

            del w_q, w_k, w_v, w_o, q_norm, k_norm, q, k, v, attn_flat
            gc.collect()

            return {
                "ple_feature": ple_features,
                "attn_out": o_out,
            }


def train_ple_and_attention_distillation(
    weights_path: str = "/run/media/akarta/AI/models/gemma-4-E2B-it/model.safetensors",
    output_dir: str = "/run/media/akarta/AI/checkpoints/ple_attention_distill",
    steps: int = 5,
    layer_idx: int = 0,
    batch_size: int = 2,
    seq_len: int = 16,
):
    os.makedirs(output_dir, exist_ok=True)
    prober = Gemma4AttentionPLEProber(weights_path)

    # Student SULA Attention Module
    student_dim = 256
    sula = SingletUncrossedAttention(
        embed_dim=student_dim,
        num_heads=8,
        state_rank=16,
        chunk_window=16,
    )

    # Alignment adapters
    # 1. Aligns Gemma 4 PLE [256] to SULA's associative rank-r projection [r*H = 128]
    ple_adapter = nn.Linear(256, student_dim, bias=False)
    # 2. Aligns SULA attention output [student_dim] to Gemma 4 attention output [1536]
    attn_adapter = nn.Linear(student_dim, 1536, bias=False)

    optimizer = torch.optim.AdamW(
        list(sula.parameters()) + list(ple_adapter.parameters()) + list(attn_adapter.parameters()),
        lr=1e-3,
    )

    print(f"=== Distilling Gemma 4 PLE & Attention Kernel for Layer {layer_idx} ===")

    for step in range(steps):
        tokens = torch.randint(0, 10000, (batch_size, seq_len))
        optimizer.zero_grad()

        # Probe Gemma 4
        teacher = prober.probe_ple_and_attention(tokens, layer_idx=layer_idx)

        # Inject distilled PLE representation into student input
        student_input = ple_adapter(teacher["ple_feature"])
        student_attn_out, _ = sula(student_input)
        projected_student_attn = attn_adapter(student_attn_out)

        # Attention Kernel Alignment Loss (MSE + Cosine)
        mse_loss = F.mse_loss(projected_student_attn, teacher["attn_out"])
        cos_loss = 1.0 - F.cosine_similarity(projected_student_attn, teacher["attn_out"], dim=-1).mean()
        total_loss = mse_loss + 0.5 * cos_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(sula.parameters(), 1.0)
        optimizer.step()

        print(f"Step {step+1}/{steps} | PLE + SULA Attention Loss: {total_loss.item():.4f} (MSE: {mse_loss.item():.4f}, Cos: {cos_loss.item():.4f})")

    save_path = os.path.join(output_dir, f"sula_ple_layer_{layer_idx}.pt")
    torch.save(sula.state_dict(), save_path)
    print(f"Distilled SULA Attention + PLE module saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()

    train_ple_and_attention_distillation(steps=args.steps, layer_idx=args.layer)
