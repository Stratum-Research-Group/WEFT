"""
WEFT Pretraining on Real-World FineWeb-Edu Stream.
Trains a WEFT model with configurable rank R on real web text.
"""

import argparse
import math
import os
import time
import torch
from weft import WeftConfig, WeftForCausalLM
from weft.dataset_fineweb import FineWebEduStreamer


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_fineweb(
    steps: int = 50,
    batch_size: int = 4,
    seq_len: int = 128,
    state_rank: int = 64,
    lr: float = 5e-4,
    save_dir: str = "weft_fineweb_ckpt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=================================================================")
    print("  WEFT PRETRAINING ON REAL-WORLD FINEWEB-EDU DATASET")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Steps: {steps} | Batch: {batch_size} | SeqLen: {seq_len} | Rank: {state_rank}")
    print("=================================================================")

    streamer = FineWebEduStreamer(seq_len=seq_len, batch_size=batch_size, device=str(device))
    vocab_size = streamer.vocab_size

    config = WeftConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_state_heads=2,
        state_rank=state_rank,
        window_size=64,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )

    model = WeftForCausalLM(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} (~{num_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("\nStarting training loop...")
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        current_lr = get_lr(step, warmup_steps=10, max_steps=steps, max_lr=lr, min_lr=lr*0.1)
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

        if step % 10 == 0 or step == steps:
            elapsed = time.time() - t0
            tok_s = (step * batch_size * seq_len) / elapsed
            print(
                f"Step {step:3d}/{steps} | LR: {current_lr:.2e} | "
                f"Train Loss: {loss.item():.4f} | Grad: {grad_norm.item():.2f} | {tok_s:.1f} tok/s"
            )

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"\nCheckpoint successfully saved to {save_dir}/")

    # Sample generation
    prompt = "Education and technology are"
    prompt_ids = torch.tensor([streamer.enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        gen = model.generate(prompt_ids, max_new_tokens=40, do_sample=True, temperature=0.8, top_k=40)
    print("\n" + "=" * 50)
    print("Generated continuation:")
    print(streamer.enc.decode(gen[0].tolist()))
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--rank", type=int, default=64)
    args = parser.parse_args()

    train_fineweb(steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len, state_rank=args.rank)
