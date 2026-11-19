"""
Large-Scale Dataset Pretraining and Validation Pipeline for WEFT.
Uses GPT-2 BPE tokenization, token packing, and AMP mixed-precision.
"""

import argparse
import math
import os
import time
import torch
import tiktoken
from weft import WeftConfig, WeftForCausalLM
from weft.dataset_loader import TokenPackedLoader


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


def train_weft(
    data_path: str = "data/tinyshakespeare.txt",
    seq_len: int = 128,
    batch_size: int = 8,
    max_steps: int = 500,
    warmup_steps: int = 50,
    lr: float = 6e-4,
    min_lr: float = 6e-5,
    save_dir: str = "weft_bpe_ckpt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=================================================================")
    print("  WEFT LARGE-SCALE PRETRAINING & VALIDATION PIPELINE")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  SeqLen: {seq_len} | Batch: {batch_size} | Steps: {max_steps}")
    print("=================================================================")

    loader = TokenPackedLoader(data_path, seq_len=seq_len, batch_size=batch_size, device=str(device))
    vocab_size = loader.vocab_size

    # Configure 50M-class WEFT Model Architecture
    config = WeftConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_state_heads=2,
        state_rank=16,
        window_size=64,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )

    model = WeftForCausalLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Mixed-precision training with: {amp_dtype}")

    print("\nStarting pretraining steps...")
    t0 = time.time()
    for step in range(1, max_steps + 1):
        model.train()
        current_lr = get_lr(step, warmup_steps, max_steps, lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        x, y = loader.get_batch("train")
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=amp_dtype):
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == max_steps:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(10):
                    vx, vy = loader.get_batch("val")
                    with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=amp_dtype):
                        vout = model(input_ids=vx, labels=vy)
                    val_losses.append(vout.loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            val_ppl = math.exp(avg_val_loss)
            elapsed = time.time() - t0
            tok_s = (step * batch_size * seq_len) / elapsed

            print(
                f"Step {step:4d}/{max_steps} | LR: {current_lr:.2e} | "
                f"Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Val PPL: {val_ppl:.2f} | Grad: {grad_norm.item():.2f} | {tok_s:.1f} tok/s"
            )

    # Save final model
    print(f"\nSaving pre-trained checkpoint to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print("Pretraining completed successfully.")

    # Generate sample text
    prompt = "The kingdom of"
    prompt_ids = torch.tensor([loader.enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    print("\n" + "=" * 50)
    print(f"Generating sample from prompt: {repr(prompt)}")
    print("=" * 50)
    with torch.no_grad():
        gen = model.generate(prompt_ids, max_new_tokens=50, do_sample=True, temperature=0.8, top_k=40)
    print(loader.enc.decode(gen[0].tolist()))
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    train_weft(max_steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len)
