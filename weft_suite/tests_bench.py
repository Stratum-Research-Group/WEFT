"""
Track A Architectural Benchmarks:
A1: Real Language Modeling Quality on WikiText split with shifted targets and held-out validation
A2: Real Multi-Query Associative Recall (MQAR) benchmark measuring exact key-value association retention
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from weft_suite.common import get_tiny_config, set_seed
from weft_suite.baselines import get_model_family

import hashlib

def get_wikitext_splits(tier="standard", val_ratio=0.2):
    """Loads WikiText-2 tokens and partitions into deterministic train/val splits.
    Fails if the file is missing outside smoke tier. Computes and returns sha256 hash.
    """
    wikitext_path = "data/wikitext2/wikitext-2-raw-test.txt"
    if os.path.exists(wikitext_path):
        with open(wikitext_path, "rb") as f:
            raw_data = f.read()
        file_hash = hashlib.sha256(raw_data).hexdigest()
        token_ids = [b + 1 for b in raw_data]
    else:
        if tier != "smoke":
            raise FileNotFoundError(
                f"Missing required real dataset {wikitext_path}. Outside smoke tier, silent fallback is prohibited. "
                f"Run 'python scripts/download_wikitext2.py' to fetch the canonical dataset."
            )
        file_hash = "synthetic_smoke_stream_fallback"
        rng = np.random.RandomState(42)
        token_ids = rng.randint(1, 257, size=50000).tolist()

    # Split into contiguous blocks (80% train / 20% val)
    split_idx = int(len(token_ids) * (1.0 - val_ratio))
    train_tokens = torch.tensor(token_ids[:split_idx], dtype=torch.long)
    val_tokens = torch.tensor(token_ids[split_idx:], dtype=torch.long)
    return train_tokens, val_tokens, file_hash

def run_a1_language_modeling_sweep(device="cpu", steps=100, seeds=(1234, 1235, 1236), tier="standard"):
    """A1: Evaluates learning curves across M1-M5 on real token streams with held-out validation."""
    print(f"[BENCH A1] Running matched architecture language modeling sweep ({device}, {steps} steps, 3 seeds, tier={tier})...")
    cfg = get_tiny_config()
    train_tokens, val_tokens, data_hash = get_wikitext_splits(tier=tier)
    seq_len = 64
    batch_size = 4
    val_steps = 20
    results = {}

    for seed in seeds:
        set_seed(seed)
        models, params_info = get_model_family(cfg, device)
        rng = np.random.RandomState(seed)

        for m_name, model in models.items():
            model.train()
            # AdamW optimizer with warmup + cosine decay
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

            # Sanity assertion on initial loss
            val_batch_starts = rng.randint(0, len(val_tokens) - seq_len - 1, size=batch_size)
            val_x = torch.stack([val_tokens[s:s+seq_len] for s in val_batch_starts]).to(device)
            val_y = torch.stack([val_tokens[s+1:s+seq_len+1] for s in val_batch_starts]).to(device)

            with torch.no_grad():
                init_logits = model(val_x).logits if hasattr(model, "model") else model(val_x)
                init_loss = F.cross_entropy(init_logits.view(-1, cfg.vocab_size), val_y.view(-1)).item()

            ln_vocab = math.log(cfg.vocab_size)
            assert abs(init_loss - ln_vocab) < 1.5, f"Initial loss {init_loss:.3f} diverged from ln(vocab) {ln_vocab:.3f}"

            train_losses = []
            for step in range(steps):
                opt.zero_grad()
                batch_starts = rng.randint(0, len(train_tokens) - seq_len - 1, size=batch_size)
                x = torch.stack([train_tokens[s:s+seq_len] for s in batch_starts]).to(device)
                y = torch.stack([train_tokens[s+1:s+seq_len+1] for s in batch_starts]).to(device)

                logits = model(x).logits if hasattr(model, "model") else model(x)
                loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                train_losses.append(loss.item())

            # Evaluate held-out validation loss over multiple batches
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(val_steps):
                    v_starts = rng.randint(0, len(val_tokens) - seq_len - 1, size=batch_size)
                    vx = torch.stack([val_tokens[s:s+seq_len] for s in v_starts]).to(device)
                    vy = torch.stack([val_tokens[s+1:s+seq_len+1] for s in v_starts]).to(device)
                    v_logits = model(vx).logits if hasattr(model, "model") else model(vx)
                    v_loss = F.cross_entropy(v_logits.view(-1, cfg.vocab_size), vy.view(-1)).item()
                    val_losses.append(v_loss)

            final_val_loss = float(np.mean(val_losses))
            final_val_ppl = float(math.exp(min(final_val_loss, 20.0)))

            if m_name not in results:
                results[m_name] = {
                    "trainable_params": params_info["trainable"][m_name],
                    "init_losses": [],
                    "final_val_losses": [],
                    "final_val_ppls": [],
                    "train_curves": []
                }
            results[m_name]["init_losses"].append(init_loss)
            results[m_name]["final_val_losses"].append(final_val_loss)
            results[m_name]["final_val_ppls"].append(final_val_ppl)
            results[m_name]["train_curves"].append(train_losses)

    summary = {"dataset_sha256": data_hash, "tier": tier, "models": {}}
    for m_name, d in results.items():
        arr_loss = np.array(d["final_val_losses"])
        arr_ppl = np.array(d["final_val_ppls"])
        summary["models"][m_name] = {
            "trainable_params": d["trainable_params"],
            "mean_val_loss": float(np.mean(arr_loss)),
            "std_val_loss": float(np.std(arr_loss)),
            "mean_val_ppl": float(np.mean(arr_ppl)),
            "std_val_ppl": float(np.std(arr_ppl)),
            "per_seed_loss": [float(v) for v in arr_loss]
        }
        print(f"  - {m_name:<22s}: Val Loss = {summary['models'][m_name]['mean_val_loss']:.4f} +- {summary['models'][m_name]['std_val_loss']:.4f} (PPL = {summary['models'][m_name]['mean_val_ppl']:.2f})")

    return summary

def generate_mqar_batch(batch_size, seq_len, num_pairs, vocab_size, device="cpu"):
    """
    Constructs real synthetic Multi-Query Associative Recall (MQAR) task batches.
    Format: k1 v1 k2 v2 ... kn vn [gap...] q_k1 q_k2 ...
    Keys sampled from [1..32], values from [33..64].
    Targets scored only at query positions.
    """
    key_min, key_max = 1, 32
    val_min, val_max = 33, 64
    num_vals = val_max - val_min + 1
    chance_level = 1.0 / float(num_vals)

    inputs = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    for b in range(batch_size):
        keys = torch.randperm(key_max - key_min + 1)[:num_pairs] + key_min
        vals = torch.randint(val_min, val_max + 1, (num_pairs,))

        # Place key-value pairs at sequence prefix
        for i in range(num_pairs):
            inputs[b, 2*i] = keys[i]
            inputs[b, 2*i + 1] = vals[i]

        # Place query tokens at sequence suffix
        query_start = max(2 * num_pairs + 4, seq_len - num_pairs)
        query_perm = torch.randperm(num_pairs)
        for qi, idx in enumerate(query_perm):
            pos = query_start + qi
            if pos < seq_len:
                inputs[b, pos] = keys[idx]
                targets[b, pos] = vals[idx]

    return inputs, targets, chance_level

def run_a2_mqar_benchmark(device="cpu", seq_len=64, num_pairs=4, steps=100, seeds=(1234, 1235, 1236)):
    """
    A2: Multi-Query Associative Recall (MQAR) benchmark testing state retention.
    Trains each model on fresh synthetic associative memory batches and evaluates exact recall on held-out tasks.
    """
    print(f"[BENCH A2] Running Multi-Query Associative Recall (SeqLen={seq_len}, Pairs={num_pairs}, {steps} steps, 3 seeds)...")
    cfg = get_tiny_config()
    batch_size = 8
    results = {}
    chance_level = None

    for seed in seeds:
        set_seed(seed)
        models, params_info = get_model_family(cfg, device)

        for m_name, model in models.items():
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)

            # Training on fresh random tasks per step
            for _ in range(steps):
                opt.zero_grad()
                inp, tgt, chance = generate_mqar_batch(batch_size, seq_len, num_pairs, cfg.vocab_size, device)
                chance_level = chance
                logits = model(inp).logits if hasattr(model, "model") else model(inp)
                loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), tgt.view(-1), ignore_index=-100)
                loss.backward()
                opt.step()

            # Evaluation on held-out fresh MQAR test tasks
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for _ in range(15):
                    inp, tgt, _ = generate_mqar_batch(batch_size, seq_len, num_pairs, cfg.vocab_size, device)
                    logits = model(inp).logits if hasattr(model, "model") else model(inp)
                    preds = torch.argmax(logits, dim=-1)
                    mask = (tgt != -100)
                    correct += (preds[mask] == tgt[mask]).sum().item()
                    total += mask.sum().item()

            acc = float(correct / max(total, 1))
            if m_name not in results:
                results[m_name] = []
            results[m_name].append(acc)

    summary = {}
    for m_name, accs in results.items():
        arr = np.array(accs)
        summary[m_name] = {
            "mean_accuracy": float(np.mean(arr)),
            "std_accuracy": float(np.std(arr)),
            "chance_level": float(chance_level) if chance_level is not None else 0.0078,
            "per_seed": [float(v) for v in arr]
        }
        print(f"  - {m_name:<22s}: Recall Acc = {summary[m_name]['mean_accuracy']:.4f} +- {summary[m_name]['std_accuracy']:.4f} (Chance = {summary[m_name]['chance_level']:.4f})")

    return summary
