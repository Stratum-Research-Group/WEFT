"""
Speculative Decoding Speedup Profiler on NVIDIA RTX 5060 Ti.
============================================================
Benchmarks speculative decoding vs. standard autoregressive baseline.
"""

import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from weft.format import WeftModelContainer, WeftSubBitLinear
from weft.serve_any import UniversalSubBitModel
from weft.speculative_engine import SpeculativeDecodingEngine


def run_speculative_speedup_benchmark(
    prompt: str = "The history of artificial intelligence began with",
    max_tokens: int = 24,
    out_path: str = "results/speculative_speedup_rtx5060ti.json",
):
    print("=== Speculative Decoding Speedup Profiler (RTX 5060 Ti) ===")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build synthetic draft and target models for clean relative speedup benchmarking
    c_draft = WeftModelContainer(metadata={"architecture": "qwen2"})
    c_target = WeftModelContainer(metadata={"architecture": "qwen2"})

    dim = 256
    vocab = 1000

    for l in range(2):
        c_draft.add_subbit_linear(f"model.layers.{l}.self_attn.q_proj.weight", WeftSubBitLinear(dim, dim, rank=16))
        c_draft.add_subbit_linear(f"model.layers.{l}.self_attn.k_proj.weight", WeftSubBitLinear(dim, dim, rank=16))
        c_draft.add_subbit_linear(f"model.layers.{l}.self_attn.v_proj.weight", WeftSubBitLinear(dim, dim, rank=16))
        c_draft.add_subbit_linear(f"model.layers.{l}.mlp.gate_proj.weight", WeftSubBitLinear(dim, dim * 2, rank=16))
        c_draft.add_subbit_linear(f"model.layers.{l}.mlp.up_proj.weight", WeftSubBitLinear(dim, dim * 2, rank=16))
        c_draft.add_subbit_linear(f"model.layers.{l}.mlp.down_proj.weight", WeftSubBitLinear(dim * 2, dim, rank=16))

        c_target.add_subbit_linear(f"model.layers.{l}.self_attn.q_proj.weight", WeftSubBitLinear(dim, dim, rank=64))
        c_target.add_subbit_linear(f"model.layers.{l}.self_attn.k_proj.weight", WeftSubBitLinear(dim, dim, rank=64))
        c_target.add_subbit_linear(f"model.layers.{l}.self_attn.v_proj.weight", WeftSubBitLinear(dim, dim, rank=64))
        c_target.add_subbit_linear(f"model.layers.{l}.mlp.gate_proj.weight", WeftSubBitLinear(dim, dim * 2, rank=64))
        c_target.add_subbit_linear(f"model.layers.{l}.mlp.up_proj.weight", WeftSubBitLinear(dim, dim * 2, rank=64))
        c_target.add_subbit_linear(f"model.layers.{l}.mlp.down_proj.weight", WeftSubBitLinear(dim * 2, dim, rank=64))

    c_draft.add_subbit_linear("model.embed_tokens.weight", WeftSubBitLinear(dim, vocab, rank=16))
    c_target.add_subbit_linear("model.embed_tokens.weight", WeftSubBitLinear(dim, vocab, rank=64))

    c_draft.save("bench_draft.weft")
    c_target.save("bench_target.weft")

    draft_model = UniversalSubBitModel(["bench_draft.weft"]).to(device)
    target_model = UniversalSubBitModel(["bench_target.weft"]).to(device)

    prompt_ids = torch.tensor([[10, 25, 42, 99]], device=device)

    # 1. Baseline Autoregressive Target Model Generation
    print("[*] Running Baseline Target Autoregressive Generation...")
    start_base = time.time()
    curr_ids = prompt_ids.clone()
    for _ in range(max_tokens):
        next_tok = target_model.generate_step(curr_ids)
        curr_ids = torch.cat([curr_ids, next_tok], dim=-1)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_base = time.time() - start_base
    base_tok_per_sec = max_tokens / max(elapsed_base, 1e-6)
    print(f"[+] Baseline: {elapsed_base:.4f} s ({base_tok_per_sec:.2f} tokens/s)")

    # 2. Speculative Decoding Sweep across lookahead gamma
    gammas = [1, 2, 3, 4, 6]
    sweep_results = []

    for g in gammas:
        engine = SpeculativeDecodingEngine(draft_model, target_model, gamma=g)
        res = engine.generate(prompt_ids, max_new_tokens=max_tokens)
        speedup = base_tok_per_sec / max(res["tokens_per_second"], 1e-6)
        print(f"Gamma={g}: {res['tokens_per_second']:.2f} tok/s, Acceptance Rate: {res['acceptance_rate']:.2f}")
        sweep_results.append({
            "gamma": g,
            "tokens_per_second": res["tokens_per_second"],
            "acceptance_rate": res["acceptance_rate"],
            "elapsed_seconds": res["elapsed_seconds"],
        })

    import os
    os.remove("bench_draft.weft")
    os.remove("bench_target.weft")

    report = {
        "device": device,
        "max_tokens": max_tokens,
        "baseline_autoregressive": {
            "elapsed_seconds": elapsed_base,
            "tokens_per_second": base_tok_per_sec,
        },
        "speculative_sweep": sweep_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[PASS] Speculative speedup results saved to {out_path}")


if __name__ == "__main__":
    run_speculative_speedup_benchmark()
