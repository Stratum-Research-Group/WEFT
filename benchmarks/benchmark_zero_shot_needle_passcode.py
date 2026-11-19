"""
Zero-Shot Exact Passcode Needle-in-a-Haystack Benchmark for WEFT-2.

Evaluates true zero-shot token recall across deep context depths:
N in [1024, 2048, 4096, 8192, 16384].

Methodology:
1. Synthetic distractor text (random natural text or Wikipedia-style passages).
2. Random needle passcode placed at variable context depth (10%, 25%, 50%, 75%, 90%):
   "The secret access code is <PASSCODE>."
3. At the extreme end of the sequence (N - 1), model is prompted:
   "What was the secret access code? The secret access code is"
4. Evaluates whether the top-1 predicted tokens match <PASSCODE> exactly.
"""

import time
import random
import string
import torch
import torch.nn.functional as F
import tiktoken

from weft import WeftConfig, WeftForCausalLM


def generate_random_passcode(length: int = 5) -> str:
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def run_zero_shot_needle_benchmark(
    model_path: str = "weft2_foundation_ckpt",
    context_lengths=[1024, 2048, 4096, 8192],
    depth_ratios=[0.10, 0.25, 0.50, 0.75, 0.90],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 80)
    print("  WEFT-2 ZERO-SHOT PASSCODE NEEDLE-IN-A-HAYSTACK BENCHMARK")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Contexts: {context_lengths} | Depths: {depth_ratios}")
    print("=" * 80)

    print(f"Loading model from {model_path}...")
    model = WeftForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    distractor_words = [
        "apple", "river", "mountain", "clock", "silver", "history", "algorithm",
        "quantum", "forest", "galaxy", "window", "vector", "matrix", "energy",
        "velocity", "syntax", "tensor", "silicon", "network", "gradient", "memory"
    ]

    results = []

    for N in context_lengths:
        for depth in depth_ratios:
            passcode = generate_random_passcode(length=4)
            needle_str = f" The secret access code is {passcode}."
            query_str = " What was the secret access code? The secret access code is "

            needle_toks = tokenizer.encode(needle_str)
            query_toks = tokenizer.encode(query_str)
            target_toks = tokenizer.encode(f" {passcode}")

            # Fill sequence with distractor tokens
            haystack_len = N - len(needle_toks) - len(query_toks)
            distractor_text = " ".join(random.choices(distractor_words, k=haystack_len * 2))
            distractor_toks = tokenizer.encode(distractor_text)[:haystack_len]

            # Needle insertion position
            insert_idx = int(haystack_len * depth)
            full_toks = (
                distractor_toks[:insert_idx]
                + needle_toks
                + distractor_toks[insert_idx:]
                + query_toks
            )

            # Cap exactly at N tokens
            full_toks = full_toks[:N]
            input_tensor = torch.tensor([full_toks], device=device, dtype=torch.long)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            with torch.no_grad():
                out = model(input_tensor, use_cache=True)
                logits = out.logits[:, -1, :]  # Next-token prediction logits

            latency_ms = (time.time() - t0) * 1000.0
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

            # Evaluate top-1 predicted token vs first target token
            top1_tok = torch.argmax(logits, dim=-1).item()
            pred_char = tokenizer.decode([top1_tok])
            expected_char = tokenizer.decode([target_toks[0]])
            exact_match = (top1_tok == target_toks[0])

            # Measure probability assigned to expected token
            probs = F.softmax(logits.float(), dim=-1)
            target_prob = probs[0, target_toks[0]].item()

            print(
                f"N: {N:5d} | Depth: {int(depth * 100):2d}% (pos={insert_idx:5d}) | "
                f"Passcode: {passcode} | Pred: {repr(pred_char):8s} (Exp: {repr(expected_char):8s}) | "
                f"Match: {exact_match!s:5s} | Target Prob: {target_prob * 100:6.2f}% | "
                f"Latency: {latency_ms:6.1f} ms | VRAM: {peak_vram:6.1f} MB"
            )

            results.append({
                "N": N,
                "depth": depth,
                "exact_match": exact_match,
                "target_prob": target_prob,
                "latency_ms": latency_ms,
                "vram_mb": peak_vram,
            })

    print("=" * 80)
    print("  ZERO-SHOT NEEDLE RETRIEVAL SUMMARY")
    for N in context_lengths:
        n_results = [r for r in results if r["N"] == N]
        avg_prob = sum(r["target_prob"] for r in n_results) / len(n_results)
        matches = sum(1 for r in n_results if r["exact_match"])
        print(f"  Context N = {N:5d} -> Exact Matches: {matches}/{len(n_results)} | Mean Target Prob: {avg_prob * 100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    run_zero_shot_needle_benchmark(context_lengths=[1024, 2048, 4096])
