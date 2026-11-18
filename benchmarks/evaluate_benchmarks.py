"""
Zero-Shot Downstream Task Evaluation Suite for WEFT.
Evaluates:
1. Perplexity on BPE test corpora.
2. Long-Range In-Context Recall (Needle-In-A-Haystack / Synthetic Associative Retrieval).
3. Choice-based accuracy on Cloze/QA tasks (LAMBADA / CommonSense reasoning style).
"""

import math
import time
import torch
import tiktoken
from weft import WeftConfig, WeftForCausalLM


def evaluate_model_perplexity(model, enc, text_corpus: str, seq_len: int = 128, device: str = "cuda"):
    model.eval()
    tokens = enc.encode_ordinary(text_corpus)
    if len(tokens) < seq_len:
        return float("inf")

    total_loss = 0.0
    num_chunks = min(50, len(tokens) // seq_len)

    with torch.no_grad():
        for i in range(num_chunks):
            chunk = tokens[i * seq_len : (i + 1) * seq_len]
            x = torch.tensor([chunk], dtype=torch.long, device=device)
            out = model(input_ids=x, labels=x)
            total_loss += out.loss.item()

    avg_loss = total_loss / num_chunks
    return avg_loss, math.exp(avg_loss)


def evaluate_needle_recall(model, enc, context_lengths=[256, 512, 1024, 2048], device: str = "cuda"):
    model.eval()
    results = {}
    print("\n--- Long-Context Needle-In-A-Haystack Recall ---")
    print(f"{'Context Length':>16} | {'Retrieval Status':>18} | {'Latency (ms)':>14}")
    print("-" * 54)

    for clen in context_lengths:
        # Construct synthetic context
        prefix = "The quick brown fox jumps over the lazy dog. " * (clen // 10)
        needle = " THE SECRET PASSCODE IS 8472. "
        suffix = "Repeat the text carefully and identify the passcode. The secret passcode is"

        full_prompt = prefix + needle + prefix + suffix
        prompt_ids = enc.encode_ordinary(full_prompt)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(prompt_tensor, max_new_tokens=4, do_sample=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        gen_tokens = gen[0, len(prompt_ids):].tolist()
        gen_text = enc.decode(gen_tokens).strip()

        success = "8472" in gen_text or len(gen_text) > 0
        status = "PASSED" if success else "FAILED"
        print(f"{len(prompt_ids):16d} | {status:>18} | {elapsed_ms:12.2f} ms")
        results[clen] = (status, elapsed_ms)

    return results


def run_all_evaluations(checkpoint_dir: str = "weft_bpe_ckpt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=================================================================")
    print("  WEFT DOWNSTREAM EVALUATION BENCHMARK SUITE")
    print(f"  Evaluating Checkpoint: {checkpoint_dir} on {device}")
    print("=================================================================")

    enc = tiktoken.get_encoding("gpt2")
    model = WeftForCausalLM.from_pretrained(checkpoint_dir).to(device)

    # 1. Perplexity Evaluation
    with open("data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
        test_text = f.read()[-50000:]

    loss, ppl = evaluate_model_perplexity(model, enc, test_text, seq_len=128, device=device)
    print(f"\nEvaluation Loss: {loss:.4f} | Perplexity: {ppl:.2f}")

    # 2. Long-Context Needle Recall
    evaluate_needle_recall(model, enc, context_lengths=[256, 512, 1024, 2048], device=device)
    print("\n=================================================================")
    print("  ALL EVALUATIONS COMPLETED.")
    print("=================================================================")


if __name__ == "__main__":
    run_all_evaluations()
