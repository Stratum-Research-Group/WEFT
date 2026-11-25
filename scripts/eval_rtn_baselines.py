"""
Item 5b: Round-To-Nearest (RTN) Quantization Baselines on Qwen2.5-0.5B-Instruct
Evaluates standard RTN uniform quantization at 4-bit, 3-bit, 2-bit, and 1.58-bit ternary
on the exact same 16 WikiText-2 windows (512 seqlen) to establish baseline reference points.
Appends baseline measurements to results/ppl_vs_bpw.json.
"""

import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from weft.dataset_loader import download_wikitext2

def quantize_rtn(W: torch.Tensor, bits: int):
    # Per-channel symmetric uniform quantization
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    abs_max = torch.max(torch.abs(W), dim=1, keepdim=True)[0].clamp_min(1e-5)
    scale = abs_max / qmax
    q = torch.clamp(torch.round(W / scale), qmin, qmax)
    return q * scale

def quantize_ternary(W: torch.Tensor):
    # Standard BitNet 1.58b: W_ternary = round(clip(W / gamma, -1, 1)) * gamma
    gamma = torch.mean(torch.abs(W), dim=1, keepdim=True).clamp_min(1e-5)
    q = torch.clamp(torch.round(W / gamma), -1.0, 1.0)
    return q * gamma

def run_baselines(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", eval_windows: int = 16, seqlen: int = 512):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[BASELINES] Running RTN Baselines on {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    test_file = download_wikitext2("test")
    with open(test_file, "r", encoding="utf-8") as f:
        text = f.read()

    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_tokens = eval_windows * (seqlen - 1)

    baselines = [
        {"name": "RTN 4-bit (Symmetric)", "bits": 4, "type": "int"},
        {"name": "RTN 3-bit (Symmetric)", "bits": 3, "type": "int"},
        {"name": "RTN 2-bit (Symmetric)", "bits": 2, "type": "int"},
        {"name": "Ternary 1.58-bit (BitNet)", "bits": 1.58, "type": "ternary"},
    ]

    baseline_results = []

    for b in baselines:
        print(f"[*] Evaluating {b['name']}...")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        for layer in model.model.layers:
            projs = [
                layer.self_attn.q_proj, layer.self_attn.k_proj,
                layer.self_attn.v_proj, layer.self_attn.o_proj,
                layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj
            ]
            for module in projs:
                orig_W = module.weight.data
                if b["type"] == "ternary":
                    q_W = quantize_ternary(orig_W)
                else:
                    q_W = quantize_rtn(orig_W, b["bits"])
                module.weight.data.copy_(q_W)

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
        ppl = float(torch.exp(torch.tensor(mean_nll)).item())
        print(f"[+] {b['name']} PPL = {ppl:.2f}")

        baseline_results.append({
            "name": b["name"],
            "bits": b["bits"],
            "ppl": ppl,
            "mean_nll": mean_nll
        })

        del model
        torch.cuda.empty_cache()

    # Append to results/ppl_vs_bpw.json
    with open("results/ppl_vs_bpw.json", "r") as f:
        data = json.load(f)
    data["rtn_baselines"] = baseline_results
    with open("results/ppl_vs_bpw.json", "w") as f:
        json.dump(data, f, indent=2)

    print("[DONE] Appended RTN baselines to results/ppl_vs_bpw.json")

if __name__ == "__main__":
    run_baselines()
