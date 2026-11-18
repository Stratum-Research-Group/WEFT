"""
Curriculum Needle-In-A-Haystack and Associative Recall Fine-Tuning for WEFT-125M.

Fine-tunes the pretrained base checkpoint (weft_125m_foundation_ckpt) on synthetic
associative key-value retrieval pairs across varying sequence lengths (256 up to 1024),
allowing the linear recurrent memory state to learn the associative retrieval operator
W_k^T W_v for long-range copying and needle recall.
"""

import os
import random
import time
import torch
import torch.nn as nn
import tiktoken
from weft import WeftForCausalLM
from weft.optimizers import Muon


def generate_needle_training_batch(batch_size: int, seq_len: int, enc, device: str = "cuda"):
    """
    Generates a batch of sequences where a secret passcode needle is embedded in distractor text,
    and the sequence ends with a query for that passcode.
    Labels are masked with -100 everywhere except the target passcode tokens.
    """
    passcodes = [f"{random.randint(1000, 9999)}" for _ in range(batch_size)]
    batch_input_ids = []
    batch_labels = []

    distractor_phrases = [
        "The quick brown fox jumps over the lazy dog. ",
        "In modern machine learning, attention mechanisms route information dynamically. ",
        "Linear recurrent models maintain an accumulated state vector over time. ",
        "The stratospheric weather patterns influence atmospheric thermodynamics. ",
        "Computational complexity scales linearly with the sequence length under WEFT. ",
    ]

    for b in range(batch_size):
        code = passcodes[b]
        needle_str = f" THE SECRET PASSCODE IS {code}. "
        query_str = " Repeat the text carefully and identify the passcode. The secret passcode is"
        target_str = f" {code}"

        needle_toks = enc.encode_ordinary(needle_str)
        query_toks = enc.encode_ordinary(query_str)
        target_toks = enc.encode_ordinary(target_str)

        # Distractor length
        remaining = seq_len - len(needle_toks) - len(query_toks) - len(target_toks)
        assert remaining > 10, f"seq_len {seq_len} is too small"

        # Split remaining into prefix and suffix
        split = random.randint(remaining // 4, 3 * remaining // 4)
        
        prefix_str = ""
        while len(enc.encode_ordinary(prefix_str)) < split:
            prefix_str += random.choice(distractor_phrases)
        prefix_toks = enc.encode_ordinary(prefix_str)[:split]

        mid_len = remaining - len(prefix_toks)
        mid_str = ""
        while len(enc.encode_ordinary(mid_str)) < mid_len:
            mid_str += random.choice(distractor_phrases)
        mid_toks = enc.encode_ordinary(mid_str)[:mid_len]

        full_toks = prefix_toks + needle_toks + mid_toks + query_toks + target_toks
        # Pad or truncate exactly to seq_len
        full_toks = full_toks[:seq_len]

        # Target labels: -100 everywhere except the final target_toks
        labels = [-100] * len(full_toks)
        t_len = len(target_toks)
        labels[-t_len:] = full_toks[-t_len:]

        batch_input_ids.append(full_toks)
        batch_labels.append(labels)

    x = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
    y = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return x, y


def train_needle_tuning(
    model_path: str = "weft_125m_foundation_ckpt",
    save_dir: str = "weft_125m_needle_ckpt",
    steps: int = 300,
    batch_size: int = 4,
    device: str = "cuda",
):
    print("=" * 70)
    print("  WEFT-125M ASSOCIATIVE NEEDLE-IN-A-HAYSTACK FINE-TUNING")
    print(f"  Base Model: {model_path} -> Output: {save_dir}")
    print(f"  Device: {device} | Steps: {steps} | Batch Size: {batch_size}")
    print("=" * 70)

    enc = tiktoken.get_encoding("gpt2")
    model = WeftForCausalLM.from_pretrained(model_path).to(device)
    model.train()
    model.gradient_checkpointing_enable()

    # Use AdamW with lower LR for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    t0 = time.time()
    for step in range(1, steps + 1):
        # Curriculum sequence length: starts at 256, expands to 512, then 1024
        if step < 100:
            seq_len = 256
        elif step < 200:
            seq_len = 512
        else:
            seq_len = 1024

        x, y = generate_needle_training_batch(batch_size, seq_len, enc, device=device)

        optimizer.zero_grad(set_to_none=True)
        out = model(x, labels=y)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0 or step == steps:
            elapsed = time.time() - t0
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Step {step:3d}/{steps} | SeqLen: {seq_len:4d} | "
                f"Needle Loss: {loss.item():.4f} | VRAM: {vram_mb:.1f} MB | "
                f"Elapsed: {elapsed:.1f}s"
            )

    print(f"\nSaving needle-tuned checkpoint to ./{save_dir}/ ...")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print("Checkpoint saved successfully.")


if __name__ == "__main__":
    train_needle_tuning()
