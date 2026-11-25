"""
Item 5: Sweep bits per weight (bpw) and rank upward on Qwen2.5-0.5B-Instruct
Evaluates WikiText-2 test perplexity across rank configurations (r=64, 128, 256, 512)
and paths (1, 2) to empirically map where PPL drops toward FP16 baseline.
Saves results to results/ppl_vs_bpw.json.
"""

import os
import json
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from weft.converter import factorize_linear_layer
from weft.dataset_loader import download_wikitext2

def run_sweep(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    eval_windows: int = 16,
    seqlen: int = 512,
    out_path: str = "results/ppl_vs_bpw.json"
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[SWEEP] Initializing Rank & BPW Sweep on {model_id} ({device})...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    test_file = download_wikitext2("test")
    with open(test_file, "r", encoding="utf-8") as f:
        text = f.read()

    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    # Evaluate pure FP16 baseline on the exact same windows
    print(f"[*] Evaluating FP16 Baseline on {eval_windows} windows...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    total_nll_fp16 = 0.0
    total_tokens = 0
    with torch.no_grad():
        for w in range(eval_windows):
            s = w * seqlen
            e = s + seqlen
            chunk = enc[s:e].unsqueeze(0).to(device)
            targets = chunk[:, 1:]
            inputs = chunk[:, :-1]
            logits = model_fp16(inputs).logits
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            total_nll_fp16 += float(loss.item())
            total_tokens += targets.numel()

    fp16_ppl = float(torch.exp(torch.tensor(total_nll_fp16 / total_tokens)).item())
    print(f"[+] FP16 PPL: {fp16_ppl:.2f}")
    del model_fp16
    torch.cuda.empty_cache()

    configs = [
        {"rank": 64, "paths": 1, "desc": "0.22 bpw (Rank 64, 1 Path)"},
        {"rank": 64, "paths": 2, "desc": "0.44 bpw (Rank 64, 2 Paths)"},
        {"rank": 128, "paths": 1, "desc": "0.45 bpw (Rank 128, 1 Path)"},
        {"rank": 256, "paths": 1, "desc": "0.89 bpw (Rank 256, 1 Path)"},
        {"rank": 512, "paths": 1, "desc": "1.79 bpw (Rank 512, 1 Path)"},
    ]

    sweep_results = {
        "model": model_id,
        "fp16_ppl": fp16_ppl,
        "eval_tokens": total_tokens,
        "eval_windows": eval_windows,
        "sweep": []
    }

    for cfg in configs:
        rank = cfg["rank"]
        paths = cfg["paths"]
        print(f"\n[*] Evaluating WEFT config: {cfg['desc']} (r={rank}, p={paths})...")

        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        layers = model.model.layers

        t0_quant = time.time()
        for layer in layers:
            projs = [
                layer.self_attn.q_proj, layer.self_attn.k_proj,
                layer.self_attn.v_proj, layer.self_attn.o_proj,
                layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj
            ]
            for module in projs:
                orig_W = module.weight.data.clone()
                subbit = factorize_linear_layer(
                    orig_W, base_rank=rank, adaptive_rank=False, num_paths=paths, itq_iterations=3
                )
                rec_W = subbit.reconstruct_weight().to(orig_W.dtype).to(device)
                module.weight.data.copy_(rec_W)

        quant_sec = time.time() - t0_quant

        total_nll = 0.0
        with torch.no_grad():
            for w in range(eval_windows):
                s = w * seqlen
                e = s + seqlen
                chunk = enc[s:e].unsqueeze(0).to(device)
                targets = chunk[:, 1:]
                inputs = chunk[:, :-1]
                logits = model(inputs).logits
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.reshape(-1))
                total_nll += float(loss.item())

        mean_nll = total_nll / total_tokens
        weft_ppl = float(torch.exp(torch.tensor(mean_nll)).item())
        print(f"[+] Result for {cfg['desc']}: PPL = {weft_ppl:.2f} (Quant time: {quant_sec:.2f}s)")

        sweep_results["sweep"].append({
            "rank": rank,
            "paths": paths,
            "desc": cfg["desc"],
            "ppl": weft_ppl,
            "mean_nll": mean_nll,
            "quant_seconds": quant_sec
        })

        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    print(f"\n[DONE] Saved sweep results to {out_path}")
    return sweep_results

if __name__ == "__main__":
    run_sweep()
