"""
WEFT-2 Chat & Multi-Turn Instruction Fine-Tuning Pipeline.

Fine-tunes the 589M WEFT-2 model on a diverse distribution of:
1. Multi-step reasoning with <think>...</think> chain-of-thought scratchpads.
2. Code generation and algorithmic problem solving (Python/Algorithms).
3. Conversational QA and knowledge recall.
4. Tool-use / JSON schema structured response adherence.
"""

import os
import time
import random
import argparse
import torch
import tiktoken
from weft import WeftForCausalLM


DIVERSE_TEMPLATES = [
    # 1. Mathematical / Arithmetic Reasoning
    lambda: (
        f"Question: Calculate ({random.randint(10, 50)} * {random.randint(4, 15)}) + {random.randint(25, 200)}.\n"
        f"Provide your reasoning in <think>...</think> tags and the exact answer after ####.\nAnswer:",
        lambda p: f" <think>Break down the multiplication and addition clearly.</think>\n#### {eval(p.split('Calculate ')[1].split('.')[0])}"
    ),
    # 2. Python Code Generation
    lambda: (
        f"Write a Python function `reverse_and_filter_{random.choice(['evens', 'odds', 'positives'])}` that takes a list of integers.",
        lambda p: (
            " Here is the clean Python implementation:\n```python\n"
            f"def {p.split('`')[1]}(nums):\n"
            "    return [x for x in reversed(nums) if x > 0]\n"
            "```"
        )
    ),
    # 3. Knowledge QA & Concept Explanation
    lambda: (
        f"Explain how {random.choice(['linear attention', 'mixture-of-experts', 'KV cache compression', 'exponential decay'])} reduces inference latency.",
        lambda p: (
            " In modern language model architectures, this mechanism replaces quadratic memory growth "
            "with constant or bounded recurrent state updates, allowing tokens to be decoded without "
            "allocating unbounded GPU high-bandwidth memory for historical sequences."
        )
    ),
    # 4. JSON Schema / Structured Tool Call
    lambda: (
        f"Extract key entity metadata as JSON for user ID 'usr_{random.randint(1000, 9999)}' with status 'active'.",
        lambda p: (
            ' ```json\n{\n'
            f'  "user_id": "{p.split("user ID ")[1].split(" ")[0]}",\n'
            '  "status": "active",\n'
            '  "verified": true\n'
            '}\n```'
        )
    ),
]


def generate_chat_sample():
    template_fn = random.choice(DIVERSE_TEMPLATES)
    prompt, comp_fn = template_fn()
    completion = comp_fn(prompt)
    return prompt, completion


def train_chat_sft(
    model_path: str = "weft2_unified_instruct_ckpt",
    save_path: str = "weft2_chat_v2_ckpt",
    steps: int = 500,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 3e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 80)
    print("  WEFT-2 EXTENDED MULTI-DOMAIN CHAT SFT PIPELINE")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Source: {model_path} -> Target: {save_path}")
    print(f"  Steps: {steps} | Batch: {batch_size} (Effective: {batch_size * grad_accum}) | LR: {lr}")
    print("=" * 80)

    print(f"Loading checkpoint from {model_path}...")
    model = WeftForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    model.train()
    model.gradient_checkpointing_enable()

    tokenizer = tiktoken.get_encoding("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, steps + 1):
        accum_loss = 0.0

        for accum_idx in range(grad_accum):
            input_ids_batch = []
            labels_batch = []

            for _ in range(batch_size):
                p, c = generate_chat_sample()
                full_text = p + c
                tokens = tokenizer.encode_ordinary(full_text)
                tokens.append(tokenizer.eot_token)

                p_len = len(tokenizer.encode_ordinary(p))
                labels = list(tokens)
                for i in range(min(p_len, len(labels))):
                    labels[i] = -100

                input_ids_batch.append(tokens)
                labels_batch.append(labels)

            max_len = max(len(t) for t in input_ids_batch)
            padded_inputs = [t + [tokenizer.eot_token] * (max_len - len(t)) for t in input_ids_batch]
            padded_labels = [l + [-100] * (max_len - len(l)) for l in labels_batch]

            x = torch.tensor(padded_inputs, device=device, dtype=torch.long)
            y = torch.tensor(padded_labels, device=device, dtype=torch.long)

            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 20 == 0 or step == steps:
            elapsed = time.time() - t0
            vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
            print(f"Step {step:4d}/{steps:4d} | Loss: {accum_loss:7.4f} | VRAM: {vram:6.1f} MB | Elapsed: {elapsed:5.1f}s")

    print(f"\nSaving final chat fine-tuned model to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print("Checkpoint saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WEFT-2 Chat SFT")
    parser.add_argument("--steps", type=int, default=300, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro batch size")
    parser.add_argument("--accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--model-path", type=str, default="weft2_unified_instruct_ckpt")
    parser.add_argument("--save-path", type=str, default="weft2_chat_v2_ckpt")
    args = parser.parse_args()

    train_chat_sft(
        model_path=args.model_path,
        save_path=args.save_path,
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.accum,
        lr=args.lr,
    )
