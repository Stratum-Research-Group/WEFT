"""
Multi-Needle In A Haystack Associative Retrieval Benchmark for WEFT-2.

Evaluates associative recall of multiple distinct key-value pairs (K >= 5)
placed at varying depths across extended contexts (16,384 to 131,072 tokens).
Validates that WEFT's associative memory and Titans surprise gate maintain
retrieval fidelity across multiple simultaneous memories without state collapse.
"""

import math
import random
import time
from typing import Dict, List, Tuple, Any
import torch
import torch.nn.functional as F
from weft.configuration_weft import WeftConfig
from weft.modeling_weft import WeftForCausalLM


def run_multi_needle_benchmark(
    checkpoint_dir: str = "weft2_release_v2_ckpt",
    context_lengths: List[int] = [16384, 32768, 65536],
    num_needles: int = 5,
    chunk_size: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"  WEFT-2 MULTI-NEEDLE HAYSTACK STRESS TEST (K={num_needles} Needles)")
    print(f"  Checkpoint: {checkpoint_dir} | Device: {device}")
    print("=" * 80)

    config = WeftConfig.from_pretrained(checkpoint_dir)
    config.use_cache = True
    model = WeftForCausalLM.from_pretrained(
        checkpoint_dir,
        config=config,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    results = []

    for ctx_len in context_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Generate background filler tokens
        torch.manual_seed(1337)
        stream_tokens = torch.randint(100, 30000, (ctx_len,), dtype=torch.long)

        # Disperse K distinct key-value pairs across the stream
        # Needle format: Key token K_i followed by Value token V_i
        needles = []
        depth_steps = ctx_len // (num_needles + 1)
        for i in range(num_needles):
            pos = depth_steps * (i + 1)
            key_id = 40000 + i
            val_id = 45000 + i
            stream_tokens[pos] = key_id
            stream_tokens[pos + 1] = val_id
            depth_pct = round((pos / ctx_len) * 100, 1)
            needles.append((key_id, val_id, pos, depth_pct))

        # Stream context in chunks through WEFT recurrent state
        input_ids = stream_tokens.unsqueeze(0).to(device)
        num_chunks = ctx_len // chunk_size
        past_state = None

        t0 = time.perf_counter()
        with torch.no_grad():
            for c in range(num_chunks):
                start = c * chunk_size
                end = start + chunk_size
                chunk = input_ids[:, start:end]
                out = model(input_ids=chunk, past_key_values=past_state, use_cache=True)
                past_state = out.past_key_values

        streaming_time = time.perf_counter() - t0
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

        # Query all K needles
        retrieved_count = 0
        needle_details = []
        for key_id, val_id, pos, depth_pct in needles:
            query_token = torch.tensor([[key_id]], device=device, dtype=torch.long)
            with torch.no_grad():
                q_out = model(input_ids=query_token, past_key_values=past_state, use_cache=True)
                pred_token = torch.argmax(q_out.logits[:, -1, :], dim=-1).item()
                # Check top-5 candidate rankings for associative match
                top5 = torch.topk(q_out.logits[:, -1, :], k=5, dim=-1).indices[0].tolist()

            matched = (pred_token == val_id) or (val_id in top5)
            if matched:
                retrieved_count += 1
            needle_details.append({
                "needle_idx": len(needle_details) + 1,
                "depth_pct": depth_pct,
                "key": key_id,
                "target_val": val_id,
                "pred_val": pred_token,
                "top5": top5,
                "hit": matched,
            })

        acc = (retrieved_count / num_needles) * 100
        print(
            f"Context: {ctx_len:6d} tokens ({num_chunks:4d} chunks) | "
            f"Recall: {retrieved_count}/{num_needles} ({acc:5.1f}%) | "
            f"VRAM: {peak_vram:6.1f} MB | Time: {streaming_time:.2f}s"
        )
        for nd in needle_details:
            hit_str = "MATCH" if nd["hit"] else "MISS"
            print(f"   -> Needle {nd['needle_idx']} @ {nd['depth_pct']:4.1f}% depth | Target: {nd['target_val']} | Top1: {nd['pred_val']} | [{hit_str}]")

        results.append({
            "context_len": ctx_len,
            "accuracy": acc,
            "retrieved": retrieved_count,
            "total_needles": num_needles,
            "peak_vram_mb": peak_vram,
            "time_sec": streaming_time,
        })

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/weft2_foundation_5k")
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384])
    args = parser.parse_args()
    run_multi_needle_benchmark(checkpoint_dir=args.checkpoint, context_lengths=args.lengths)
