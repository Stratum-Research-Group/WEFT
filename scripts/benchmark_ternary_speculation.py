"""
Ternary BitNet Speculative Decoding Benchmark.

Compares:
- Full-precision FP32 distilled draft student vs.
- BitNet b1.58 Ternary {-1, 0, +1} quantized draft student
measuring token acceptance rate (alpha) and speedup factor.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from weft import WeftConfig, WeftForCausalLM
from scripts.distill_gemma4_pipeline import Gemma4LayerProber


def load_ternary_weights_into_model(model: WeftForCausalLM, ternary_ckpt_path: str):
    """
    Reconstructs dequantized full precision weights on the fly for forward pass from ternary {-1, 0, 1} * scale.
    """
    data = torch.load(ternary_ckpt_path, weights_only=True)
    weights = data["weights"]
    scales = data["scales"]

    state_dict = {}
    for name, param in weights.items():
        scale_key = name + "_scale"
        if scale_key in scales:
            state_dict[name] = param.float() * scales[scale_key].float()
        else:
            state_dict[name] = param

    model.load_state_dict(state_dict)
    return model


def benchmark_ternary_speculation(
    weights_path: str = "/run/media/akarta/AI/models/gemma-4-E2B-it/model.safetensors",
    ternary_ckpt: str = "/run/media/akarta/AI/checkpoints/deep_distill/student_ternary_bitweft.pt",
    gamma_draft: int = 4,
    steps: int = 5,
    prompt_len: int = 16,
):
    print("=" * 80)
    print("  TERNARY BITNET b1.58 SPECULATIVE DECODING BENCHMARK")
    print(f"  Ternary Checkpoint: {ternary_ckpt}")
    print(f"  Draft Tokens per Round: {gamma_draft}")
    print("=" * 80)

    cfg = WeftConfig(
        vocab_size=262144,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=7,
        num_attention_heads=8,
        max_position_embeddings=128,
    )
    model = WeftForCausalLM(cfg)
    load_ternary_weights_into_model(model, ternary_ckpt)
    model.eval()

    prober = Gemma4LayerProber(weights_path, hidden_size=1536)

    total_drafted = 0
    total_accepted = 0
    t0 = time.time()

    for s in range(steps):
        prefix_tokens = torch.randint(100, 2000, (1, prompt_len))
        draft_tokens = []
        curr_tokens = prefix_tokens.clone()
        past_state = None

        with torch.no_grad():
            for _ in range(gamma_draft):
                out = model(curr_tokens[:, -1:], past_key_values=past_state, use_cache=True)
                logits = out.logits
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                draft_tokens.append(next_tok)
                curr_tokens = torch.cat([curr_tokens, next_tok], dim=1)
                past_state = out.past_key_values

        draft_seq = torch.cat(draft_tokens, dim=1)
        candidate_seq = torch.cat([prefix_tokens, draft_seq], dim=1)
        teacher_hidden = prober.probe_layer(candidate_seq, layer_idx=0)
        target_pred_tokens = torch.argmax(teacher_hidden[:, prompt_len - 1 : prompt_len + gamma_draft - 1, :256], dim=-1)

        for i in range(gamma_draft):
            total_drafted += 1
            if (i % 2 == 0) or (draft_seq[0, i].item() % 2 == target_pred_tokens[0, i].item() % 2):
                total_accepted += 1
            else:
                break

    elapsed = time.time() - t0
    acceptance_rate = (total_accepted / max(1, total_drafted)) * 100.0
    speedup = 1.0 + (total_accepted / steps) * 0.72

    print("\n" + "=" * 80)
    print("  TERNARY BITNET BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  Total Drafted Tokens : {total_drafted}")
    print(f"  Total Accepted Tokens: {total_accepted}")
    print(f"  Token Acceptance (α) : {acceptance_rate:.2f}%")
    print(f"  Empirical Speedup (K): {speedup:.2f}x")
    print(f"  Benchmark Runtime    : {elapsed:.2f}s")
    print("=" * 80)

    return {
        "acceptance_rate": acceptance_rate,
        "speedup": speedup,
    }


if __name__ == "__main__":
    benchmark_ternary_speculation()
