"""
WEFT-2 Native Tool-Calling SFT Bootstrapper.

Trains WEFT-2 to parse system tool schemas and generate syntactically valid
JSON tool calls enclosed in `<tool_call> ... </tool_call>` tags, as well as
resuming reasoning once a `<tool_response> ... </tool_response>` is provided.
"""

import os
import sys
import time
import random
import json
import torch
import tiktoken
from weft import WeftForCausalLM
from weft.tool_calling import ToolRegistry


def build_synthetic_tool_scenario():
    """Generates synthetic multi-turn tool calling dialogue with system schemas."""
    tools = [
        {
            "name": "get_stock_price",
            "desc": "Fetch real-time stock ticker price and currency.",
            "params": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
            "examples": [
                ("AAPL", 225.40, "USD"),
                ("MSFT", 418.15, "USD"),
                ("NVDA", 142.80, "USD"),
                ("GOOGL", 175.20, "USD"),
            ],
        },
        {
            "name": "calculate_mortgage",
            "desc": "Compute monthly mortgage payment given principal, annual rate, and years.",
            "params": {
                "type": "object",
                "properties": {
                    "principal": {"type": "number"},
                    "rate_percent": {"type": "number"},
                    "years": {"type": "integer"},
                },
                "required": ["principal", "rate_percent", "years"],
            },
            "examples": [
                (400000, 6.5, 30, 2528.27),
                (250000, 5.0, 15, 1977.06),
            ],
        },
    ]

    choice = random.choice(tools)
    if choice["name"] == "get_stock_price":
        ticker, price, curr = random.choice(choice["examples"])
        sys_prompt = f"You are a helpful assistant with access to tools.\nTools available:\n- get_stock_price: {choice['desc']}\n"
        user_msg = f"User: Can you check the current stock price of {ticker}?\nAssistant:"
        call_json = json.dumps({"name": "get_stock_price", "arguments": {"ticker": ticker}})
        tool_call = f" <tool_call>\n{call_json}\n</tool_call>"
        resp_json = json.dumps({"status": "success", "ticker": ticker, "price": price, "currency": curr})
        tool_resp = f"\n<tool_response>\n{resp_json}\n</tool_response>\nAssistant:"
        final_ans = f" The current price of {ticker} is {price} {curr}."
    else:
        p, r, y, monthly = random.choice(choice["examples"])
        sys_prompt = f"You are a helpful assistant with access to tools.\nTools available:\n- calculate_mortgage: {choice['desc']}\n"
        user_msg = f"User: Calculate the monthly payment for a ${p:,} loan at {r}% for {y} years.\nAssistant:"
        call_json = json.dumps({"name": "calculate_mortgage", "arguments": {"principal": p, "rate_percent": r, "years": y}})
        tool_call = f" <tool_call>\n{call_json}\n</tool_call>"
        resp_json = json.dumps({"status": "success", "monthly_payment": monthly})
        tool_resp = f"\n<tool_response>\n{resp_json}\n</tool_response>\nAssistant:"
        final_ans = f" Based on the calculation, the monthly mortgage payment is ${monthly:.2f}."

    prompt = sys_prompt + user_msg
    completion = tool_call + tool_resp + final_ans
    return prompt, completion


def train_weft2_tool_sft(
    model_path: str = "checkpoints/weft2_code_sft",
    save_path: str = "checkpoints/weft2_tool_sft",
    steps: int = 40,
    batch_size: int = 4,
    lr: float = 3e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 80)
    print("  WEFT-2 AGENTIC TOOL-CALLING SFT")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Base Model: {model_path} | Save Target: {save_path}")
    print(f"  Steps: {steps} | Batch: {batch_size} | LR: {lr}")
    print("=" * 80)

    actual_path = model_path if os.path.exists(model_path) else "weft2_release_v2_ckpt"
    model = WeftForCausalLM.from_pretrained(actual_path, torch_dtype=torch.bfloat16).to(device)
    model.train()
    model.gradient_checkpointing_enable()

    tokenizer = tiktoken.get_encoding("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    t0 = time.time()
    for step in range(1, steps + 1):
        input_ids_batch = []
        labels_batch = []

        for _ in range(batch_size):
            p, c = build_synthetic_tool_scenario()
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
        padded_inputs = []
        padded_labels = []
        for t, l in zip(input_ids_batch, labels_batch):
            pad_len = max_len - len(t)
            padded_inputs.append(t + [tokenizer.eot_token] * pad_len)
            padded_labels.append(l + [-100] * pad_len)

        x = torch.tensor(padded_inputs, dtype=torch.long, device=device)
        y = torch.tensor(padded_labels, dtype=torch.long, device=device)

        optimizer.zero_grad()
        out = model(x, labels=y)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 10 == 0 or step == steps or step == 1:
            tok_per_sec = (step * batch_size * max_len) / (time.time() - t0)
            print(f"Step {step:3d}/{steps} | Loss: {loss.item():.4f} | Throughput: {tok_per_sec:.0f} tok/s")

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Saved Tool SFT model to {save_path}")


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    train_weft2_tool_sft(steps=steps)
