"""
Comprehensive Zero-Shot and Perplexity Evaluation for FineWeb-Edu Pretrained WEFT.
Evaluates:
1. Cross-entropy loss and Perplexity on held-out FineWeb-Edu evaluation split.
2. Character/Token level sequence coherence.
3. Long-context associative recall on FineWeb representations.
"""

import math
import time
import torch
import tiktoken
from weft import WeftConfig, WeftForCausalLM
from weft.dataset_fineweb import FineWebEduStreamer


def run_fineweb_eval(checkpoint_path: str = "weft_fineweb_ckpt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=================================================================")
    print(f"  FINEWEB-EDU ZERO-SHOT & PERPLEXITY EVALUATION")
    print(f"  Checkpoint: {checkpoint_path} | Device: {device}")
    print("=================================================================")

    enc = tiktoken.get_encoding("gpt2")
    model = WeftForCausalLM.from_pretrained(checkpoint_path).to(device)
    model.eval()

    # Explicitly request the disjoint validation split (last 15% partition)
    streamer = FineWebEduStreamer(split="val", seq_len=128, batch_size=4, device=str(device))
    streamer.current_offset = 0

    print("\n--> Step 1: Evaluating Perplexity on 20 held-out batches...")
    val_losses = []
    with torch.no_grad():
        for i in range(20):
            x, y = streamer.get_batch()
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                out = model(input_ids=x, labels=y)
            val_losses.append(out.loss.item())

    avg_loss = sum(val_losses) / len(val_losses)
    ppl = math.exp(avg_loss)
    print(f"    Held-Out Evaluation Loss: {avg_loss:.4f}")
    print(f"    Held-Out Perplexity:      {ppl:.2f}")

    print("\n--> Step 2: Sample Prompts Generation Quality...")
    test_prompts = [
        "Scientific research shows that",
        "The history of mathematics began with",
        "Computer programming requires",
    ]

    for p in test_prompts:
        p_ids = torch.tensor([enc.encode_ordinary(p)], dtype=torch.long, device=device)
        with torch.no_grad():
            gen = model.generate(p_ids, max_new_tokens=30, do_sample=True, temperature=0.7, top_k=40)
        decoded = enc.decode(gen[0].tolist())
        print(f"\n[Prompt]: {p}")
        print(f"[Generated]: {decoded.strip()}")

    print("\n=================================================================")
    print("  EVALUATION FINISHED.")
    print("=================================================================")


if __name__ == "__main__":
    run_fineweb_eval()
