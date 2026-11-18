"""
WEFT-2 Unified Foundation Model Pretraining Pipeline.

Architecture:
- Layers: 12
- Hidden Dimension: 768
- Attention Heads: 12 (Sliding Window W=128)
- Local Path: Multi-Head Latent Attention (MLA) (d_c=128)
- Global Recurrence: Dual-Axis Memory Initialization (DAMI) timescales (64 to 65,536)
- Test-Time Associative Plasticity: Titans Surprise-Gated Memory (d_mem=64)
- FFN Mixer: DeepSeekMoE (1 shared expert + 8 routed experts, top-2 active)
- Optimizer: Muon (Newton-Schulz Orthogonalized Momentum) with AMP (bfloat16)
- Dataset: FineWeb-Edu Stream (GPT-2 BPE, 50,257 vocab)
"""

import argparse
import math
import os
import time
import torch
from weft import WeftConfig, WeftForCausalLM
from weft.dataset_fineweb import FineWebEduStreamer
from weft.optimizers import Muon


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float):
    if warmup_steps >= max_steps:
        warmup_steps = max(1, max_steps // 5)
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_weft2_foundation(
    steps: int = 10000,
    batch_size: int = 4,
    seq_len: int = 128,
    max_lr: float = 0.02,
    min_lr: float = 0.002,
    save_dir: str = "weft2_foundation_ckpt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  WEFT-2 UNIFIED FOUNDATION PRETRAINING PIPELINE")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Steps: {steps} | Batch: {batch_size} | SeqLen: {seq_len} | Precision: bfloat16")
    print("=" * 80)

    # Initialize FineWeb-Edu streamer
    streamer = FineWebEduStreamer(seq_len=seq_len, batch_size=batch_size, device=str(device))
    vocab_size = streamer.vocab_size

    # Configure WEFT-2 model with MLA, DAMI, Titans, and DeepSeekMoE
    config = WeftConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_state_heads=4,
        state_rank=64,
        window_size=128,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        shifted_binding=True,
        use_mla=True,
        mla_latent_dim=128,
        use_titans=True,
        titans_mem_dim=64,
        use_moe=True,
        moe_num_routed_experts=8,
        moe_top_k=2,
    )

    model = WeftForCausalLM(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture: 12 Layers | 768 Dim | MLA(dc=128) | Titans(64) | DeepSeekMoE(8 experts, top-2)")
    print(f"Total Parameters: {num_params:,} (~{num_params/1e6:.1f}M)")

    # Muon optimizer
    optimizer = Muon(model.parameters(), lr=max_lr, momentum=0.95, weight_decay=0.01)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"Optimizer: Muon (~2 bytes/param state overhead)")
    print(f"Beginning clean ground-up pretraining run...\n")

    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        current_lr = get_lr(step, warmup_steps=100, max_steps=steps, max_lr=max_lr, min_lr=min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        x, y = streamer.get_batch()
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=amp_dtype):
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss

            # Add auxiliary load-balancing loss across MoE layers
            aux_loss = 0.0
            for layer in model.model.layers:
                if hasattr(layer.mlp, "get_aux_loss"):
                    aux_loss = aux_loss + layer.mlp.get_aux_loss()

            total_loss = loss + 0.01 * aux_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0 or step == steps:
            elapsed = time.time() - t0
            tok_s = (step * batch_size * seq_len) / elapsed
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            print(
                f"Step {step:5d}/{steps} | LR: {current_lr:.2e} | "
                f"Loss: {loss.item():.4f} | Aux: {aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss:.3f} | "
                f"Grad: {grad_norm.item():.2f} | VRAM: {vram_mb:.1f} MB | {tok_s:.1f} tok/s"
            )

        if step % 1000 == 0 or step == steps:
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            print(f"  [Checkpoint] Saved WEFT-2 model to {save_dir}/ at step {step}")

    print(f"\nPretraining finished. Final WEFT-2 checkpoint saved to ./{save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--save-dir", type=str, default="weft2_foundation_ckpt")
    args = parser.parse_args()

    train_weft2_foundation(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        save_dir=args.save_dir,
    )
