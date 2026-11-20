"""
WEFT-2 Extended Pretraining Daemon (Multi-Domain Mixture).

Runs continuous pretraining blending:
1. Natural Language Reasoning (FineWeb-Edu)
2. Python & Algorithmic Code Synthesis (Synthetic AST pairs)
3. Multi-Turn Agentic Tool Trajectories

Optimized with:
- Muon (Newton-Schulz) orthogonalized momentum optimizer
- Cosine LR schedule with warmup
- Gradient checkpointing enabled
- Low VRAM footprint (< 1.2 GB VRAM)
- Checkpoint auto-saving every 2,500 steps
"""

import argparse
import json
import os
import signal
import sys
import time
import random
import torch
import tiktoken
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from weft.configuration_weft import WeftConfig
from weft.modeling_weft import WeftForCausalLM
from weft.dataset_fineweb import FineWebEduStreamer
from weft.optimizers import Muon
from examples.train_weft2_code_sft import generate_code_sample
from examples.train_weft2_tool_sft import build_synthetic_tool_scenario


STOP_REQUESTED = False


def sigterm_handler(signum, frame):
    global STOP_REQUESTED
    print("\n[INFO] Termination signal received. Flushing state and saving emergency checkpoint...")
    STOP_REQUESTED = True


signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigterm_handler)


def get_cosine_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + torch.cos(torch.tensor(3.1415926535 * progress)).item())


def main():
    parser = argparse.ArgumentParser(description="WEFT-2 Extended Multi-Domain Pretraining Daemon")
    parser.add_argument("--steps", type=int, default=50000, help="Total pretraining steps")
    parser.add_argument("--warmup", type=int, default=500, help="Warmup steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--max_lr", type=float, default=0.012, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=0.0005, help="Min learning rate")
    parser.add_argument("--save_interval", type=int, default=2500, help="Steps between checkpoints")
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between log writes")
    parser.add_argument("--resume_from", type=str, default="checkpoints/weft2_foundation_15k", help="Base checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/weft2_extended_50k")
    parser.add_argument("--metrics_file", type=str, default="artifacts/extended_pretrain_convergence.json")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_file) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 80)
    print("  WEFT-2 EXTENDED MULTI-DOMAIN PRETRAINING DAEMON")
    print(f"  Target: {args.steps} steps | Device: {device} | Precision: {dtype}")
    print(f"  Base Checkpoint: {args.resume_from}")
    print(f"  Mixture: 50% FineWeb-Edu, 30% Synthetic Code AST, 20% Agentic Tool Trajectories")
    print(f"  Checkpoint Dir: {args.checkpoint_dir} (every {args.save_interval} steps)")
    print("=" * 80)

    if os.path.exists(args.resume_from):
        print(f"Resuming weights from {args.resume_from}...")
        model = WeftForCausalLM.from_pretrained(args.resume_from, torch_dtype=dtype).to(device)
    else:
        print("Initializing standard 8-layer base model...")
        cfg = WeftConfig(
            vocab_size=50257,
            hidden_size=512,
            intermediate_size=1536,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_state_heads=4,
            state_dim=64,
            state_rank=16,
            chunk_size=32,
        )
        model = WeftForCausalLM(cfg).to(device=device, dtype=dtype)

    model.train()
    model.gradient_checkpointing_enable()

    tokenizer = tiktoken.get_encoding("gpt2")
    streamer = FineWebEduStreamer(batch_size=args.batch_size, seq_len=args.seq_len)

    optimizer = Muon(model.parameters(), lr=args.max_lr, momentum=0.95, weight_decay=0.01)

    metrics_log = []
    t0 = time.time()
    tokens_processed = 0

    print(f"Pretraining commenced on {device}...")

    for step in range(1, args.steps + 1):
        if STOP_REQUESTED:
            break

        # Adjust LR via cosine schedule
        lr = get_cosine_lr(step, args.warmup, args.steps, args.max_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Sample multi-domain mixture
        roll = random.random()
        if roll < 0.50:
            # 50% FineWeb-Edu continuous text stream
            batch = streamer.get_batch()
            input_ids, labels = batch if isinstance(batch, tuple) else (batch["input_ids"], batch["labels"])
            input_ids = input_ids.to(device)
            labels = labels.to(device)
        elif roll < 0.80:
            # 30% Python Algorithmic Code AST
            batch_inputs, batch_labels = [], []
            for _ in range(args.batch_size):
                p, c = generate_code_sample()
                tokens = tokenizer.encode_ordinary(p + c)[:args.seq_len]
                pad_len = args.seq_len - len(tokens)
                batch_inputs.append(tokens + [tokenizer.eot_token] * pad_len)
                batch_labels.append(tokens + [-100] * pad_len)
            input_ids = torch.tensor(batch_inputs, dtype=torch.long, device=device)
            labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
        else:
            # 20% Multi-Turn Agentic Tool Calls
            batch_inputs, batch_labels = [], []
            for _ in range(args.batch_size):
                p, c = build_synthetic_tool_scenario()
                tokens = tokenizer.encode_ordinary(p + c)[:args.seq_len]
                pad_len = args.seq_len - len(tokens)
                batch_inputs.append(tokens + [tokenizer.eot_token] * pad_len)
                batch_labels.append(tokens + [-100] * pad_len)
            input_ids = torch.tensor(batch_inputs, dtype=torch.long, device=device)
            labels = torch.tensor(batch_labels, dtype=torch.long, device=device)

        optimizer.zero_grad()
        out = model(input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        tokens_processed += input_ids.numel()

        if step % args.log_interval == 0 or step == 1 or step == args.steps:
            elapsed = time.time() - t0
            tok_per_sec = tokens_processed / max(1e-5, elapsed)
            vram_mb = torch.cuda.memory_allocated() / (1024**2) if device.type == "cuda" else 0
            log_entry = {
                "step": step,
                "loss": round(loss.item(), 4),
                "lr": round(lr, 6),
                "tok_s": round(tok_per_sec, 1),
                "vram_mb": round(vram_mb, 1),
                "elapsed_s": round(elapsed, 1),
            }
            metrics_log.append(log_entry)
            print(f"Step {step:6d}/{args.steps} | Loss: {loss.item():.4f} | LR: {lr:.6f} | Throughput: {tok_per_sec:.0f} tok/s | VRAM: {vram_mb:.1f} MB")

            with open(args.metrics_file, "w") as f:
                json.dump(metrics_log, f, indent=2)

        if step % args.save_interval == 0 or step == args.steps:
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step}")
            model.save_pretrained(ckpt_path)
            print(f"[CHECKPOINT] Saved step {step} to {ckpt_path}")

    # Final save
    final_dir = os.path.join(args.checkpoint_dir, "final")
    model.save_pretrained(final_dir)
    print(f"\n[DONE] Pretraining completed. Final checkpoint saved to {final_dir}")


if __name__ == "__main__":
    main()
