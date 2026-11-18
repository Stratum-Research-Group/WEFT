"""
Generalized In-Context Copying & Associative Needle Curriculum Tuning.

Forces the model to learn dynamic copying (induction) rather than memorizing a constant passcode.
Every sequence has a completely random 4-digit code (0% fixed constants), mixed with language modeling loss.
"""

import os
import random
import time
import torch
import torch.nn as nn
import tiktoken
from weft import WeftForCausalLM


def generate_generalized_copy_batch(batch_size: int, seq_len: int, enc, device: str = "cuda"):
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
    ]

    for b in range(batch_size):
        # 100% random 4-digit code: MUST copy dynamically from context
        code = f"{random.randint(1000, 9999)}"

        needle_str = f" THE SECRET PASSCODE IS {code}. "
        query_str = "Repeat the text carefully and identify the passcode. The secret passcode is"
        target_str = f" {code}"

        needle_toks = enc.encode_ordinary(needle_str)
        query_toks = enc.encode_ordinary(query_str)
        target_toks = enc.encode_ordinary(target_str)

        distractor_budget = seq_len - len(needle_toks) - len(query_toks) - len(target_toks)
        split = random.randint(distractor_budget // 4, 3 * distractor_budget // 4)

        prefix_str = ""
        while len(enc.encode_ordinary(prefix_str)) < split:
            prefix_str += random.choice(distractor_sentences)
        prefix_toks = enc.encode_ordinary(prefix_str)[:split]

        mid_len = distractor_budget - len(prefix_toks)
        mid_str = ""
        while len(enc.encode_ordinary(mid_str)) < mid_len:
            mid_str += random.choice(distractor_sentences)
        mid_toks = enc.encode_ordinary(mid_str)[:mid_len]

        full_toks = prefix_toks + needle_toks + mid_toks + query_toks + target_toks
        full_toks = full_toks[:seq_len]

        labels = [-100] * len(full_toks)
        t_len = len(target_toks)
        labels[-t_len:] = full_toks[-t_len:]

        batch_input_ids.append(full_toks)
        batch_labels.append(labels)

    x = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
    y = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return x, y


def train_generalized_induction(
    model_path: str = "weft_125m_foundation_ckpt",
    save_dir: str = "weft_125m_general_needle_ckpt",
    steps: int = 500,
    batch_size: int = 4,
    device: str = "cuda",
):
    print("=" * 75)
    print("  WEFT-125M GENERALIZED IN-CONTEXT INDUCTION TUNING")
    print(f"  Source: {model_path} -> Output: {save_dir}")
    print(f"  Steps: {steps} | Batch: {batch_size} | 100% Random Unseen Passcodes")
    print("=" * 75)

    enc = tiktoken.get_encoding("gpt2")
    model = WeftForCausalLM.from_pretrained(model_path).to(device)
    model.train()
    model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    t0 = time.time()
    for step in range(1, steps + 1):
        if step <= 150:
            seq_len = 256
        elif step <= 300:
            seq_len = 512
        else:
            seq_len = 1024

        x, y = generate_generalized_copy_batch(batch_size, seq_len, enc, device=device)

        optimizer.zero_grad(set_to_none=True)
        out = model(x, labels=y)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == steps:
            elapsed = time.time() - t0
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Step {step:4d}/{steps} | SeqLen: {seq_len:4d} | "
                f"Induction Loss: {loss.item():.4f} | VRAM: {vram_mb:.1f} MB | "
                f"Elapsed: {elapsed:.1f}s"
            )

    print(f"\nSaving generalized induction checkpoint to ./{save_dir}/ ...")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print("Checkpoint saved successfully.")


if __name__ == "__main__":
    train_generalized_induction()
