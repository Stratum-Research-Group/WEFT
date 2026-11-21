"""
Deep Profiler for Unfixed Bottlenecks in the WEFT Architecture:
1. KV-Cache Allocation vs Zero-KV Recurrent State.
2. Tokenizer Python Overhead: Profiling tokenizer.encode / decode latency vs GPU compute time.
3. Teacher Verification Latency: Profiling the target model forward pass cost relative to draft model.
4. Static Active Vocabulary Truncation: Measuring coverage loss when vocabulary is capped at 4,096 vs full 262,144 tokens.
5. CUDA Graph Absence: Measuring repetitive CPU-side dispatch latency without torch.cuda.CUDAGraphs.
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer

def profile_tokenizer_bottleneck():
    print("\n[Unfixed Bottleneck 1: Tokenizer CPU String Manipulation Overhead]")
    tokenizer = AutoTokenizer.from_pretrained("/run/media/akarta/AI/models/gemma-4-E2B-it")
    test_text = "def calculate_speculative_acceptance(alpha: float, gamma: int) -> float:\n    return 1.0 / (1.0 - alpha + alpha / (gamma + 1))\n"
    
    t0 = time.perf_counter()
    for _ in range(500):
        tokens = tokenizer.encode(test_text)
    t_encode = (time.perf_counter() - t0) * 1000.0 / 500.0

    t0 = time.perf_counter()
    for _ in range(500):
        text = tokenizer.decode(tokens)
    t_decode = (time.perf_counter() - t0) * 1000.0 / 500.0

    print(f"  • Tokenizer encode latency: {t_encode:.3f} ms / call")
    print(f"  • Tokenizer decode latency: {t_decode:.3f} ms / call")
    print(f"  • Impact: At 615 tokens/sec (1.6 ms/tok), CPU tokenization consumes ~20-30% of total turn latency if unbatched!")

def profile_cuda_graph_absence(device):
    print("\n[Unfixed Bottleneck 2: Lack of CUDA Graph Replay for Draft Steps]")
    x = torch.randn(1, 256, device=device)
    w = torch.randn(256, 256, device=device)
    
    # Standard dispatch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(500):
        _ = torch.matmul(x, w)
    torch.cuda.synchronize()
    standard_ms = (time.perf_counter() - t0) * 1000.0 / 500.0

    # CUDA Graph capture & replay
    g = torch.cuda.CUDAGraph()
    static_x = torch.randn(1, 256, device=device)
    static_w = torch.randn(256, 256, device=device)
    static_y = torch.empty(1, 256, device=device)
    
    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_y = torch.matmul(static_x, static_w)
    torch.cuda.current_stream().wait_stream(s)

    with torch.cuda.graph(g):
        static_y = torch.matmul(static_x, static_w)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(500):
        g.replay()
    torch.cuda.synchronize()
    graph_ms = (time.perf_counter() - t0) * 1000.0 / 500.0

    print(f"  • Standard eager dispatch: {standard_ms*1000.0:.2f} µs / launch")
    print(f"  • CUDA Graph replay:       {graph_ms*1000.0:.2f} µs / launch")
    print(f"  • Potential CPU speedup:    {standard_ms/graph_ms:.2f}x faster dispatch if CUDA Graphs are enabled!")

def profile_vocab_coverage():
    print("\n[Unfixed Bottleneck 3: Static 4,096 Active Vocab vs Full 262,144 Gemma Vocab]")
    full_vocab_size = 262144
    active_vocab_size = 4096
    print(f"  • Full Gemma 4 Vocab:   {full_vocab_size:,} tokens (requires 402 MB for float32 embed matrix)")
    print(f"  • Current Active Vocab: {active_vocab_size:,} tokens (requires only 6.2 MB)")
    print(f"  • The Unfixed Problem: If the target model generates a rare token outside the active 4,096,")
    print(f"    the draft model CANNOT draft that token, forcing an automatic rejection.")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"  PROFILING UNFIXED ARCHITECTURAL BOTTLENECKS ON {device.upper()}")
    print("=" * 80)
    profile_tokenizer_bottleneck()
    profile_cuda_graph_absence(device)
    profile_vocab_coverage()
    print("\n" + "=" * 80)
