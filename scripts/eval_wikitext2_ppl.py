#!/usr/bin/env python3
"""
Real WikiText-2 perplexity: FP16 baseline vs WEFT-factorized weights.

Measured, not hardcoded. Protocol:
  * test split of WikiText-2 (raw), rows joined with "\\n\\n"
  * tokenised once with the model's own tokenizer
  * non-overlapping windows of --seqlen tokens (no stride overlap), loss summed over
    all predicted tokens, PPL = exp(total_nll / total_predicted_tokens)
  * identical windows for baseline and WEFT run

WEFT variant: every nn.Linear inside model.model.layers is replaced by the dense
reconstruction of its WeftSubBitLinear factorization (factorize_linear_layer).
Embeddings, lm_head and norms stay FP16 -- this deviation is recorded in the output.
Raw results are written as JSON so they can be inspected/re-checked.
"""
import argparse, json, math, os, platform, time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from weft.converter import factorize_linear_layer
from weft.dataset_loader import download_wikitext2


@torch.no_grad()
def wikitext_ppl(model, input_ids: torch.Tensor, seqlen: int, device) -> dict:
    n_windows = input_ids.numel() // seqlen
    total_nll, total_tok = 0.0, 0
    for w in range(n_windows):
        x = input_ids[w * seqlen:(w + 1) * seqlen].unsqueeze(0).to(device)
        logits = model(x).logits.float()
        nll = nn.functional.cross_entropy(
            logits[0, :-1], x[0, 1:], reduction="sum"
        )
        total_nll += nll.item()
        total_tok += x.numel() - 1
    return {
        "windows": n_windows,
        "predicted_tokens": total_tok,
        "mean_nll": total_nll / total_tok,
        "ppl": math.exp(total_nll / total_tok),
    }


def weftize_(model, base_rank, itq_iterations, num_paths) -> dict:
    n_layers, total_params, total_bits, sq_err, sq_ref = 0, 0, 0.0, 0.0, 0.0
    for lname, layer in model.model.layers.named_modules():
        pass
    for parent_name, parent in list(model.model.layers.named_modules()):
        for cname, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                W = child.weight.data.float()
                wl = factorize_linear_layer(
                    W, base_rank=base_rank, num_paths=num_paths,
                    itq_iterations=itq_iterations,
                )
                with torch.no_grad():
                    Wr = wl.reconstruct_weight().to(W.device).float()
                    sq_err += (W - Wr).pow(2).sum().item()
                    sq_ref += W.pow(2).sum().item()
                    child.weight.data.copy_(Wr.to(child.weight.dtype))
                n_layers += 1
                total_params += W.numel()
                total_bits += wl.bits_per_weight() * W.numel()
                del wl, Wr
    return {
        "linear_layers_replaced": n_layers,
        "linear_params": total_params,
        "effective_bits_per_weight": total_bits / max(total_params, 1),
        "relative_frobenius_error": math.sqrt(sq_err / max(sq_ref, 1e-30)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--base-rank", type=int, default=64)
    ap.add_argument("--num-paths", type=int, default=1)
    ap.add_argument("--itq-iterations", type=int, default=3)
    ap.add_argument("--max-windows", type=int, default=0, help="0 = full test split")
    ap.add_argument("--out", default="results/wikitext2_ppl.json")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    text = open(download_wikitext2("test"), encoding="utf-8").read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    if a.max_windows:
        ids = ids[: a.max_windows * a.seqlen]
    print(f"[data] {ids.numel():,} tokens, seqlen={a.seqlen}")

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to(device).eval()

    t0 = time.time()
    base = wikitext_ppl(model, ids, a.seqlen, device)
    base["seconds"] = time.time() - t0
    print(f"[FP16 ] PPL = {base['ppl']:.4f}")

    t0 = time.time()
    qinfo = weftize_(model, a.base_rank, a.itq_iterations, a.num_paths)
    qinfo["quantize_seconds"] = time.time() - t0
    t0 = time.time()
    weft = wikitext_ppl(model, ids, a.seqlen, device)
    weft["seconds"] = time.time() - t0
    print(f"[WEFT ] PPL = {weft['ppl']:.4f}  (bpw={qinfo['effective_bits_per_weight']:.4f})")

    out = {
        "model": a.model,
        "protocol": "wikitext-2-raw test, '\\n\\n'-joined, non-overlapping windows",
        "seqlen": a.seqlen,
        "weft_config": {"base_rank": a.base_rank, "num_paths": a.num_paths,
                        "itq_iterations": a.itq_iterations,
                        "kept_fp16": ["embed_tokens", "lm_head", "norms"]},
        "fp16": base, "weft": weft, "quantization": qinfo,
        "ppl_delta": weft["ppl"] - base["ppl"],
        "ppl_ratio": weft["ppl"] / base["ppl"],
        "env": {"torch": torch.__version__, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "python": platform.python_version()},
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()
