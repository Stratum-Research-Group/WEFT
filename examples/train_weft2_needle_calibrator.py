"""
High-Precision Extended-Horizon Key-Value Associative Needle Calibrator for WEFT-2 (589M).

Calibrates the linear recurrent memory state, shifted binding, and Titans associative memory
to perform exact in-context key-value retrieval and induction copying across deep context horizons (512 -> 4096).
"""

import os
import random
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from weft import WeftForCausalLM


ENTITIES = [
    " Apple", " Banana", " Cherry", " Orange", " Grape", " Lemon", " Peach", " Berry",
    " Diamond", " Silver", " Gold", " Copper", " Bronze", " Iron", " Steel", " Ruby",
    " Python", " Cobra", " Viper", " Falcon", " Eagle", " Hawk", " Tiger", " Lion",
    " Tokyo", " Paris", " Berlin", " Rome", " London", " Madrid", " Dublin", " Cairo",
    " Matrix", " Tensor", " Vector", " Scalar", " Quantum", " Photon", " Laser", " Proton",
    " Rocket", " Comet", " Planet", " Star", " Moon", " Sun", " Galaxy",
    " Phoenix", " Dragon"
]

DISTRACTORS = [
    " The quick brown fox jumps over the lazy dog.",
    " Windowed attention handles high-frequency local syntactic interactions.",
    " Continuous exponential decay compresses historical features into a factorized matrix.",
    " Tensor rank uncrossing enables parallel linear-time chunk state scans.",
    " Associative recall is solved through bilinear key and value projections.",
    " Machine learning models process natural language via continuous embeddings.",
    " GPU memory bandwidth is conserved by avoiding dense quadratic matrices.",
    " DeepSeekMoE dynamically balances specialized expert computation across tokens.",
    " Titans surprise-gated fast weights adapt representations at inference time.",
    " Dual-axis memory initialization guarantees uniform multi-scale spectral coverage.",
    " The weather in the morning was remarkably quiet and serene across the hills.",
    " Computational linguistics utilizes statistical language representations for semantic parsing.",
]


def generate_extended_needle_batch(batch_size: int, seq_len: int, enc, entity_tokens, device: str = "cuda"):
    batch_input_ids = []
    batch_labels = []

    for b in range(batch_size):
        target_tok = random.choice(entity_tokens)
        target_word = enc.decode([target_tok])

        # Dual needle patterns to enforce invariant induction
        if random.random() < 0.5:
            needle_str = f" The designated secret code is{target_word}."
            query_str = " What was the designated secret code? The designated secret code is"
        else:
            needle_str = f" Note that the special item is{target_word}."
            query_str = " Question: what is the special item? Answer: the special item is"

        needle_toks = enc.encode_ordinary(needle_str)
        query_toks = enc.encode_ordinary(query_str)

        budget = seq_len - len(needle_toks) - len(query_toks) - 1
        if budget < 10:
            raise ValueError(f"seq_len {seq_len} is too short")

        depth = random.uniform(0.10, 0.90)
        split = int(budget * depth)

        prefix_words = []
        while len(enc.encode_ordinary("".join(prefix_words))) < split:
            prefix_words.append(random.choice(DISTRACTORS))
        prefix_toks = enc.encode_ordinary("".join(prefix_words))[:split]

        suffix_len = budget - len(prefix_toks)
        suffix_words = []
        while len(enc.encode_ordinary("".join(suffix_words))) < suffix_len:
            suffix_words.append(random.choice(DISTRACTORS))
        suffix_toks = enc.encode_ordinary("".join(suffix_words))[:suffix_len]

        full_input = prefix_toks + needle_toks + suffix_toks + query_toks + [target_tok]
        full_input = full_input[:seq_len]

        labels = [-100] * len(full_input)
        labels[-1] = target_tok

        batch_input_ids.append(full_input)
        batch_labels.append(labels)

    return (
        torch.tensor(batch_input_ids, dtype=torch.long, device=device),
        torch.tensor(batch_labels, dtype=torch.long, device=device),
    )


def train_extended_needle_calibrator(
    model_path: str = "weft2_foundation_ckpt",
    save_dir: str = "weft2_needle_calibrated_ckpt",
    steps: int = 200,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 80)
    print("  WEFT-2 (589M) EXTENDED-HORIZON KEY-VALUE NEEDLE CALIBRATOR")
    print(f"  Source: {model_path} -> Target: {save_dir}")
    print(f"  Device: {device} | Total Steps: {steps} | Base LR: {lr}")
    print("=" * 80)

    enc = tiktoken.get_encoding("gpt2")
    entity_tokens = [enc.encode_ordinary(w)[0] for w in ENTITIES if len(enc.encode_ordinary(w)) == 1]
    print(f"Loaded {len(entity_tokens)} atomic single-token retrieval targets.")

    model = WeftForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.train()
    model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    t0 = time.time()
    for step in range(1, steps + 1):
        # Staged progressive curriculum across context horizons:
        # Step 1-60:    512 tokens (B=8)
        # Step 61-120:  1024 tokens (B=4, grad_accum=2)
        # Step 121-170: 2048 tokens (B=2, grad_accum=4)
        # Step 171-200: 4096 tokens (B=1, grad_accum=8)
        if step <= 60:
            seq_len = 512
            micro_batch = 8
            grad_accum = 1
        elif step <= 120:
            seq_len = 1024
            micro_batch = 4
            grad_accum = 2
        elif step <= 170:
            seq_len = 2048
            micro_batch = 2
            grad_accum = 4
        else:
            seq_len = 4096
            micro_batch = 1
            grad_accum = 8

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(grad_accum):
            x, y = generate_extended_needle_batch(micro_batch, seq_len, enc, entity_tokens, device=device)
            out = model(x, labels=y)
            loss = out.loss / grad_accum
            loss.backward()
            accum_loss += loss.item() * grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0 or step == steps:
            elapsed = time.time() - t0
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            print(
                f"Step {step:4d}/{steps} | SeqLen: {seq_len:4d} | "
                f"Loss: {accum_loss:.4f} | VRAM: {vram_mb:.1f} MB | "
                f"Elapsed: {elapsed:.1f}s"
            )

    print(f"\nSaving extended needle-calibrated checkpoint to ./{save_dir}/ ...")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print("Checkpoint saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="weft2_foundation_ckpt")
    parser.add_argument("--save-dir", type=str, default="weft2_needle_calibrated_ckpt")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train_extended_needle_calibrator(
        model_path=args.model_path,
        save_dir=args.save_dir,
        steps=args.steps,
        lr=args.lr,
    )
