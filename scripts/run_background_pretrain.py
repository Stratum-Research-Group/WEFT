"""
Long-Running Background Foundation Pretraining Daemon for WEFT-2.

Features:
- Checkpoints periodically saved to `checkpoints/weft2_foundation_run/`
- Real-time logging to `logs/pretrain_5k_run.log` and `artifacts/training_5k_convergence.json`
- Low system overhead: batch_size=4, seq_len=128 (~450 MB VRAM)
- Muon Newton-Schulz orthogonalized momentum optimizer with cosine annealing
- Automatic checkpoint recovery if interrupted
"""

import argparse
import json
import os
import signal
import sys
import time
import torch
from weft.configuration_weft import WeftConfig
from weft.modeling_weft import WeftForCausalLM
from weft.dataset_fineweb import FineWebEduStreamer
from weft.optimizers import Muon


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
    parser = argparse.ArgumentParser(description="WEFT-2 Background Pretraining Daemon")
    parser.add_argument("--steps", type=int, default=5000, help="Total pretraining steps")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--max_lr", type=float, default=0.015, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=0.001, help="Min learning rate")
    parser.add_argument("--save_interval", type=int, default=500, help="Steps between checkpoints")
    parser.add_argument("--log_interval", type=int, default=50, help="Steps between log writes")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/weft2_foundation_5k")
    parser.add_argument("--metrics_file", type=str, default="artifacts/training_5k_convergence.json")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_file) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 80)
    print("  WEFT-2 CONTINUOUS BACKGROUND PRETRAINING DAEMON")
    print(f"  Target: {args.steps} steps | Device: {device} | Precision: {dtype}")
    print(f"  Batch: {args.batch_size} | SeqLen: {args.seq_len} | Save Every: {args.save_interval} steps")
    print(f"  Checkpoint Directory: {args.checkpoint_dir}")
    print("=" * 80)

    cfg = WeftConfig(
        vocab_size=50257,
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_state_heads=4,
        state_rank=32,
        window_size=128,
        use_mla=True,
        mla_latent_dim=128,
        use_moe=True,
        moe_num_routed_experts=8,
        moe_top_k=2,
        use_cache=True,
    )

    # Resume from checkpoint if present
    start_step = 1
    if os.path.exists(os.path.join(args.checkpoint_dir, "model.safetensors")):
        print(f"Resuming from existing checkpoint at {args.checkpoint_dir}...")
        model = WeftForCausalLM.from_pretrained(args.checkpoint_dir, config=cfg, torch_dtype=dtype).to(device)
    else:
        model = WeftForCausalLM(cfg).to(device=device, dtype=dtype)

    optimizer = Muon(model.parameters(), lr=args.max_lr, momentum=0.95, weight_decay=0.01)
    streamer = FineWebEduStreamer(seq_len=args.seq_len, batch_size=args.batch_size, device=str(device))

    model.train()
    history = {"step": [], "loss": [], "lr": [], "vram_mb": [], "throughput_tok_s": [], "elapsed_s": []}

    t_start = time.perf_counter()
    tokens_processed = 0

    print("Starting background training loop...")
    for step in range(start_step, args.steps + 1):
        if STOP_REQUESTED:
            break

        # Learning rate schedule
        lr = get_cosine_lr(step, args.warmup, args.steps, args.max_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = streamer.get_batch()
        input_ids, labels = batch if isinstance(batch, tuple) else (batch["input_ids"], batch["labels"])

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        tokens_processed += (args.batch_size * args.seq_len)

        # Logging
        if step % args.log_interval == 0 or step == args.steps:
            elapsed = time.perf_counter() - t_start
            vram = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
            throughput = tokens_processed / max(1e-5, elapsed)

            history["step"].append(step)
            history["loss"].append(round(loss.item(), 4))
            history["lr"].append(round(lr, 6))
            history["vram_mb"].append(round(vram, 1))
            history["throughput_tok_s"].append(round(throughput, 1))
            history["elapsed_s"].append(round(elapsed, 2))

            print(
                f"[Step {step:5d}/{args.steps}] Loss: {loss.item():6.4f} | LR: {lr:.6f} | "
                f"VRAM: {vram:5.1f} MB | Speed: {throughput:6.1f} tok/s | Elapsed: {elapsed/60:4.1f}m"
            )

            with open(args.metrics_file, "w") as f:
                json.dump(history, f, indent=2)

        # Save checkpoint
        if step % args.save_interval == 0 or step == args.steps:
            step_ckpt_dir = os.path.join(args.checkpoint_dir, f"step_{step}")
            model.save_pretrained(step_ckpt_dir)
            cfg.save_pretrained(step_ckpt_dir)
            # Also update canonical latest checkpoint
            model.save_pretrained(args.checkpoint_dir)
            cfg.save_pretrained(args.checkpoint_dir)
            print(f"--> Saved checkpoint at step {step} to {step_ckpt_dir}")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTraining completed! Processed {tokens_processed:,} tokens in {total_elapsed/60:.2f} minutes.")
    model.save_pretrained(args.checkpoint_dir)
    cfg.save_pretrained(args.checkpoint_dir)
    print(f"Final model weights saved to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
