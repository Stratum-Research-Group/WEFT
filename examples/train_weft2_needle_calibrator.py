"""
High-Precision Key-Value Associative Needle Calibrator for WEFT-2 (589M).

Calibrates the linear recurrent memory state, shifted binding, and Titans associative memory
to perform exact in-context key-value retrieval and induction copying across long context horizons (512 -> 4096).
"""

import os
import random
import time
import string
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from weft import WeftForCausalLM


def generate_needle_batch(batch_size: int, seq_len: int, enc, device: str = "cuda"):
    batch_input_ids = []
    batch_labels = []

    distractor_sentences = [
        "The quick brown fox jumps over the lazy dog. ",
        "Windowed attention handles high-frequency local syntactic interactions. ",
        "Continuous exponential decay compresses historical features into a factorized matrix. ",
        "Tensor rank uncrossing enables parallel linear-time chunk state scans. ",
        "Associative recall is solved through bilinear key and value projections. ",
        "Machine learning models process natural language via continuous embeddings. ",
        "GPU memory bandwidth is conserved by avoiding dense quadratic matrices. ",
        "DeepSeekMoE dynamically balances specialized expert computation across tokens. ",
        "Titans surprise-gated fast weights adapt representations at inference time. ",
    ]

    for b in range(batch_size):
        # Generate random alphanumeric passcode
        code_chars = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
        code = f" {code_chars}"

        needle_str = f" The secret access code is{code}."
        query_str = " What was the secret access code? The secret access code is"
        target_str = code

        needle_toks = enc.encode_ordinary(needle_str)
        query_toks = enc.encode_ordinary(query_str)
        target_toks = enc.encode_ordinary(target_str)

        distractor_budget = seq_len - len(needle_toks) - len(query_toks) - len(target_toks)
        if distractor_budget < 10:
            raise ValueError(f"seq_len {seq_len} is too short for needle and query")

        # Insertion depth: random depth between 10% and 90%
        depth = random.uniform(0.10, 0.90)
        split = int(distractor_budget * depth)

        # Assemble distractor prefix and suffix
        prefix_words = []
        while len(enc.encode_ordinary(" ".join(prefix_words))) < split:
            prefix_words.append(random.choice(distractor_sentences))
        prefix_toks = enc.encode_ordinary(" ".join(prefix_words))[:split]

        suffix_len = distractor_budget - len(prefix_toks)
        suffix_words = []
        while len(enc.encode_ordinary(" ".join(suffix_words))) < suffix_len:
            suffix_words.append(random.choice(distractor_sentences))
        suffix_toks = enc.encode_ordinary(" ".join(suffix_words))[:suffix_len]

        full_input = prefix_toks + needle_toks + suffix_toks + query_toks + target_toks
        # Pad or trim to exact seq_len
        full_input = full_input[:seq_len]

        # Labels: mask out all tokens with -100 except the target tokens at the query end
        labels = [-100] * len(full_input)
        target_start = len(prefix_toks) + len(needle_toks) + len(suffix_toks) + len(query_toks)
        for idx in range(target_start, min(target_start + len(target_toks), len(full_input))):
            labels[idx] = full_input[idx]

        batch_input_ids.append(full_input)
        batch_labels.append(labels)

    return (
        torch.tensor(batch_input_ids, dtype=torch.long, device=device),
        torch.tensor(batch_labels, dtype=torch.long, device=device),
    )


def train_needle_calibrator(
    model_path: str = "weft2_foundation_ckpt",
    save_dir: str = "weft2_needle_calibrated_ckpt",
    steps: int = 150,
    batch_size: int = 8,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 80)
    print("  WEFT-2 (589M) ASSOCIATIVE NEEDLE & KEY-VALUE IN-CONTEXT CALIBRATOR")
    print(f"  Source: {model_path} -> Target: {save_dir}")
    print(f"  Device: {device} | Steps: {steps} | Batch: {batch_size} | LR: {lr}")
    print("=" * 80)

    enc = tiktoken.get_encoding("gpt2")
    model = WeftForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.train()
    model.gradient_checkpointing_enable()

    # Train with AdamW lr=3e-4 to calibrate associative induction heads
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    t0 = time.time()
    for step in range(1, steps + 1):
        # Curriculum: 256 -> 512 -> 1024 -> 2048
        if step <= 40:
            seq_len = 256
        elif step <= 80:
            seq_len = 512
        elif step <= 120:
            seq_len = 1024
        else:
            seq_len = 2048

        x, y = generate_needle_batch(batch_size, seq_len, enc, device=device)

        optimizer.zero_grad(set_to_none=True)
        out = model(x, labels=y)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0 or step == steps:
            elapsed = time.time() - t0
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            print(
                f"Step {step:4d}/{steps} | SeqLen: {seq_len:4d} | "
                f"Loss: {loss.item():.4f} | VRAM: {vram_mb:.1f} MB | "
                f"Elapsed: {elapsed:.1f}s"
            )

    print(f"\nSaving needle-calibrated checkpoint to ./{save_dir}/ ...")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print("Checkpoint saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="weft2_foundation_ckpt")
    parser.add_argument("--save-dir", type=str, default="weft2_needle_calibrated_ckpt")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train_needle_calibrator(
        model_path=args.model_path,
        save_dir=args.save_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
