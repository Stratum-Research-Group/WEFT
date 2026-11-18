"""
WEFT-125M Foundation Model Pretraining Pipeline.
Architecture:
- Parameters: ~128.2 Million
- Layers: 12
- Hidden Dimension: 768
- Attention Heads: 12
- State Heads: 4
- State Rank: 64
- Window Size: 128
- Associative Memory Weights (AMW): Enabled (num_slots=512, rank=64)
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


def train_125m_foundation(
    steps: int = 1000,
    batch_size: int = 4,
    seq_len: int = 128,
    max_lr: float = 0.02,
    min_lr: float = 0.002,
    save_dir: str = "weft_125m_foundation_ckpt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=========================================================================")
    print("  WEFT-125M FOUNDATION MODEL PRETRAINING")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Steps: {steps} | Batch: {batch_size} | SeqLen: {seq_len} | Precision: bfloat16")
    print("=========================================================================")

    # Initialize streamer
    streamer = FineWebEduStreamer(seq_len=seq_len, batch_size=batch_size, device=str(device))
    vocab_size = streamer.vocab_size

    # Configure 125M-parameter WEFT-AMW model
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
        use_amw=True,
        amw_num_slots=512,
        amw_rank=64,
    )

    model = WeftForCausalLM(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Architecture: 12 Layers, 768 Hidden Dim, AMW Slots 512")
    print(f"Total Parameters:   {num_params:,} (~{num_params/1e6:.1f}M)")

    # Muon optimizer: Newton-Schulz orthogonalization on 2D weight matrices (2 bytes/param)
    optimizer = Muon(model.parameters(), lr=max_lr, momentum=0.95, weight_decay=0.01)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"Optimizer: Muon (~2 bytes/param state overhead)")
    print(f"Starting foundation pretraining run...\n")

    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        current_lr = get_lr(step, warmup_steps=50, max_steps=steps, max_lr=max_lr, min_lr=min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        x, y = streamer.get_batch()
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=amp_dtype):
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0 or step == steps:
            elapsed = time.time() - t0
            tok_s = (step * batch_size * seq_len) / elapsed
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            print(
                f"Step {step:4d}/{steps} | LR: {current_lr:.2e} | "
                f"Loss: {loss.item():.4f} | Grad: {grad_norm.item():.2f} | "
                f"VRAM: {vram_mb:.1f} MB | {tok_s:.1f} tok/s"
            )

        if step % 250 == 0:
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)

    # Save foundation checkpoint
    print(f"\nSaving WEFT-125M foundation checkpoint to ./{save_dir}/ ...")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print("Checkpoint saved successfully.")

    # Generate sample
    prompt = "Artificial intelligence and human reasoning"
    prompt_ids = torch.tensor([streamer.enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        gen = model.generate(prompt_ids, max_new_tokens=40, do_sample=True, temperature=0.8, top_k=40)
    print("\n" + "=" * 50)
    print(f"Sample generation from: {repr(prompt)}")
    print("=" * 50)
    print(streamer.enc.decode(gen[0].tolist()))
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    train_125m_foundation(steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len)
