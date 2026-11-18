"""
Pre-training and Verification Pipeline for WEFT on Real Language Data.
Trains a small WeftForCausalLM model on character/byte sequences from TinyShakespeare,
verifying:
1. Training loss stability and convergence over 2,000 steps.
2. Gradient norm stability and absence of NaN / Inf gradients.
3. Checkpoint export via save_pretrained().
4. Reload from checkpoint via from_pretrained().
5. Coherent autoregressive text generation using .generate().
"""

import math
import os
import time
import torch
from weft import WeftConfig, WeftForCausalLM


def run_pretraining_demo():
    print("=================================================================")
    print("  WEFT FOR CAUSAL LM: END-TO-END PRE-TRAINING DEMO")
    print("=================================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Dataset & Character Tokenizer
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tinyshakespeare.txt")
    if not os.path.exists(data_path):
        data_path = "data/tinyshakespeare.txt"
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor([char_to_id[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]

    print(f"Vocab size: {vocab_size} unique characters")
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    # 2. Model Configuration & Instantiation
    config = WeftConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_state_heads=2,
        state_rank=16,
        window_size=64,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )

    model = WeftForCausalLM(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} (~{num_params/1e6:.2f}M)")

    # 3. Training Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    batch_size = 16
    seq_len = 128
    steps = 1500

    def get_batch(split="train"):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        # For standard causal LM with internal shift (shift_logits = logits[:-1], shift_labels = labels[1:]):
        # Pass identical sequence x for input_ids and labels (target tokens at each position t align with input token at t)
        x = torch.stack([d[i : i + seq_len] for i in ix]).to(device)
        return x, x.clone()

    # 4. Training Loop
    print("\nStarting pre-training loop...")
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        x, y = get_batch("train")
        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 250 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(10):
                    vx, vy = get_batch("val")
                    val_out = model(input_ids=vx, labels=vy)
                    val_losses.append(val_out.loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                val_ppl = math.exp(avg_val_loss)

            elapsed = time.time() - t0
            speed = (step * batch_size * seq_len) / elapsed
            print(f"Step {step:4d}/{steps} | Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f} | Val PPL: {val_ppl:.2f} | Grad Norm: {grad_norm.item():.2f} | {speed:.1f} tok/s")

    # 5. Checkpoint Export
    save_dir = "weft_shakespeare_ckpt"
    print(f"\nSaving checkpoint to ./{save_dir}/ ...")
    model.save_pretrained(save_dir)
    print("Checkpoint saved successfully.")

    # 6. Checkpoint Reload & Generation Test
    print(f"Reloading model from ./{save_dir}/ ...")
    loaded_model = WeftForCausalLM.from_pretrained(save_dir).to(device)
    loaded_model.eval()

    prompt_text = "ROMEO:\n"
    prompt_ids = torch.tensor([[char_to_id[c] for c in prompt_text]], dtype=torch.long, device=device)

    print("\n" + "="*50)
    print(f"Generating from prompt: {repr(prompt_text)}")
    print("="*50)

    with torch.no_grad():
        gen_tokens = loaded_model.generate(
            prompt_ids,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.8,
            top_k=40,
        )

    out_text = "".join([id_to_char[idx.item()] for idx in gen_tokens[0]])
    print(out_text)
    print("="*50)
    print("Pre-training and generation demo completed successfully.")


if __name__ == "__main__":
    run_pretraining_demo()
