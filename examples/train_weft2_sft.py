"""
WEFT-2 Supervised Fine-Tuning (SFT) Format Cold-Start Bootstrapper.

Trains the 589M WEFT-2 foundation model on synthetic arithmetic and algorithmic
scratchpad formatting pairs (<think>...</think> -> #### [answer]) for 200 steps
to provide the warm start required by RLVR / GRPO policy optimization.
"""

import os
import time
import random
import torch
import tiktoken
from weft import WeftForCausalLM


def generate_sft_sample():
    a = random.randint(3, 30)
    b = random.randint(3, 30)
    c = random.randint(2, 9)
    d = random.randint(1, 30)

    step1 = a + b
    step2 = step1 * c
    final_ans = step2 - d

    prompt = (
        f"Question: Compute (({a} + {b}) * {c}) - {d}.\n"
        f"Think step by step inside <think>...</think> and write the exact final integer answer as #### [number].\n"
        f"Answer:"
    )
    completion = (
        f" <think>\n"
        f"1. First, compute inside parentheses: {a} + {b} = {step1}.\n"
        f"2. Next, multiply by {c}: {step1} * {c} = {step2}.\n"
        f"3. Finally, subtract {d}: {step2} - {d} = {final_ans}.\n"
        f"</think>\n"
        f"#### {final_ans}"
    )
    return prompt, completion


def train_weft2_sft(
    model_path: str = "weft2_foundation_ckpt",
    save_path: str = "weft2_sft_ckpt",
    steps: int = 200,
    batch_size: int = 4,
    lr: float = 5e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 80)
    print("  WEFT-2 FORMAT BOOTSTRAP SFT PIPELINE")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Steps: {steps} | Batch: {batch_size} | LR: {lr}")
    print("=" * 80)

    print(f"Loading base checkpoint from {model_path}...")
    model = WeftForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    model.train()
    model.gradient_checkpointing_enable()

    tokenizer = tiktoken.get_encoding("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    t0 = time.time()
    for step in range(1, steps + 1):
        input_ids_batch = []
        labels_batch = []

        for _ in range(batch_size):
            p, c = generate_sft_sample()
            full_text = p + c
            tokens = tokenizer.encode_ordinary(full_text)
            tokens.append(tokenizer.eot_token)

            # Mask out prompt tokens from loss computation
            p_len = len(tokenizer.encode_ordinary(p))
            labels = list(tokens)
            for i in range(min(p_len, len(labels))):
                labels[i] = -100

            input_ids_batch.append(tokens)
            labels_batch.append(labels)

        # Pad batch to max length in batch
        max_len = max(len(t) for t in input_ids_batch)
        padded_inputs = [t + [tokenizer.eot_token] * (max_len - len(t)) for t in input_ids_batch]
        padded_labels = [l + [-100] * (max_len - len(l)) for l in labels_batch]

        x = torch.tensor(padded_inputs, device=device, dtype=torch.long)
        y = torch.tensor(padded_labels, device=device, dtype=torch.long)

        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0 or step == steps:
            elapsed = time.time() - t0
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            print(
                f"Step {step:4d}/{steps} | "
                f"SFT Loss: {loss.item():6.4f} | "
                f"VRAM: {vram_mb:6.1f} MB | Elapsed: {elapsed:5.1f}s"
            )

    print(f"\nSFT training complete. Saving warm-start checkpoint to {save_path}/...")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Model saved successfully to {save_path}/.")


if __name__ == "__main__":
    train_weft2_sft(steps=200)
