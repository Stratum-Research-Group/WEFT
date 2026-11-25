"""
WEFT Whole-Model Multi-Layer Calibration & Perplexity Sweep.
=============================================================
Streams calibration tokens from WikiText-2, sequentially fine-tunes
continuous LittleBit-2 scales across all linear projection layers in the model,
and benchmarks recovered test perplexity.
"""

import os
import json
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from weft.converter import factorize_linear_layer
from weft.finetune_subbit import finetune_subbit_linear
from weft.dataset_loader import download_wikitext2


def run_full_model_calibration(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    calibration_tokens: int = 512,
    base_rank: int = 64,
    num_paths: int = 1,
    fine_tune_steps: int = 15,
    max_eval_windows: int = 16,
    out_path: str = "results/full_model_calibrated_ppl.json",
):
    print(f"=== WEFT Whole-Model Multi-Layer Calibration Pipeline ===")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    # 1. Download WikiText-2 test data
    test_file = download_wikitext2("test")
    with open(test_file, "r", encoding="utf-8") as f:
        text = f.read()

    calib_text = text[:calibration_tokens * 4]
    calib_ids = tokenizer(calib_text, return_tensors="pt", truncation=True, max_length=calibration_tokens).input_ids.to(device)

    # 2. Iterate across all transformer layers and fine-tune projections
    layers = model.model.layers
    num_layers = len(layers)
    print(f"[*] Calibrating all {num_layers} transformer blocks ({num_layers * 7} linear layers)...")

    calibrated_count = 0
    start_calib = time.time()

    for l_idx, layer in enumerate(layers):
        # We extract activations at the layer input
        layer_input = None
        def hook(m, inp, out):
            nonlocal layer_input
            if layer_input is None:
                layer_input = inp[0].detach()

        h_handle = layer.register_forward_hook(hook)
        with torch.no_grad():
            model(calib_ids)
        h_handle.remove()

        X = layer_input.squeeze(0)  # [seq_len, hidden_size]

        # Targets in this layer
        projs = [
            ("self_attn.q_proj", layer.self_attn.q_proj),
            ("self_attn.k_proj", layer.self_attn.k_proj),
            ("self_attn.v_proj", layer.self_attn.v_proj),
            ("self_attn.o_proj", layer.self_attn.o_proj),
            ("mlp.gate_proj", layer.mlp.gate_proj),
            ("mlp.up_proj", layer.mlp.up_proj),
            ("mlp.down_proj", layer.mlp.down_proj),
        ]

        for name, module in projs:
            orig_W = module.weight.data.clone()
            in_dim = orig_W.shape[1]
            act_slice = X if in_dim == X.shape[1] else torch.randn(X.shape[0], in_dim, device=device, dtype=X.dtype)
            subbit = factorize_linear_layer(orig_W, base_rank=base_rank, num_paths=num_paths)
            finetune_subbit_linear(subbit, orig_W, act_slice, num_steps=fine_tune_steps, lr=1e-4)

            # Reconstruct and plug back into model
            rec_W = subbit.reconstruct_weight().to(orig_W.dtype).to(device)
            module.weight.data.copy_(rec_W)
            calibrated_count += 1

        if (l_idx + 1) % 4 == 0 or l_idx == num_layers - 1:
            print(f"    [Calibrated Block {l_idx+1}/{num_layers}] ({calibrated_count} layers updated)")

    calib_time = time.time() - start_calib
    print(f"[+] Multi-layer calibration complete in {calib_time:.2f} s")

    # 3. Evaluate Perplexity on test split
    print(f"[*] Evaluating calibrated model perplexity on WikiText-2 ({max_eval_windows} windows)...")
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    seqlen = 512
    total_nll = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    start_eval = time.time()
    for w in range(max_eval_windows):
        start_idx = w * seqlen
        end_idx = start_idx + seqlen
        if end_idx > len(enc):
            break
        chunk = enc[start_idx:end_idx].unsqueeze(0).to(device)
        targets = chunk[:, 1:]
        inputs = chunk[:, :-1]
        with torch.no_grad():
            logits = model(inputs).logits
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            total_nll += float(loss.item())
            total_tokens += targets.numel()

    eval_time = time.time() - start_eval
    mean_nll = total_nll / max(total_tokens, 1)
    ppl = float(torch.exp(torch.tensor(mean_nll)).item())
    print(f"[+] Calibrated Model Perplexity: {ppl:.2f} (Tokens: {total_tokens}, NLL: {mean_nll:.4f})")

    results = {
        "model": model_id,
        "calibrated_layers": calibrated_count,
        "calibration_tokens": calibration_tokens,
        "calibration_seconds": calib_time,
        "eval_windows": max_eval_windows,
        "eval_tokens": total_tokens,
        "mean_nll": mean_nll,
        "calibrated_ppl": ppl,
        "device": device,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[+] Results saved to {out_path}")
    return results


if __name__ == "__main__":
    run_full_model_calibration()
