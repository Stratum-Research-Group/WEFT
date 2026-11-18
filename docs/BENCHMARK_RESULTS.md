# Empirical Benchmark: Structured Block-Plus-Low-Rank vs. PyTorch SDPA

**Date:** 2026-11-18  
**Hardware:** NVIDIA GeForce RTX 5060 Ti (16 GB VRAM)  
**Environment:** PyTorch 2.15.0+cu132, CUDA 13.2, Python 3.13  
**Implementation:** [benchmark_attention.py](../benchmarks/benchmark_attention.py)

---

## 1. Experimental Setup

We implemented an empirical benchmark to test Hypothesis A:
- **Baseline:** `torch.nn.functional.scaled_dot_product_attention` (FlashAttention/cuDNN kernel).
- **Candidate:** `StructuredAttention` combining:
  1. Local sliding window multi-head attention (window size $w = 256$, complexity $O(N \cdot w \cdot d)$).
  2. Global low-rank projection bottleneck (rank $r = 32$, complexity $O(N \cdot r \cdot d)$).
- **Model Parameters:** $d_{\text{model}} = 1024$, $n_{\text{heads}} = 16$, `batch_size = 1`, `fp16` precision.
- **Sequence Lengths Tested:** $N \in \{1024, 2048, 4096, 8192, 16384, 32768\}$.

---

## 2. Empirical Results

| Sequence Length ($N$) | PyTorch SDPA Latency | SDPA Peak Mem (MB) | Structured Latency | Structured Peak Mem (MB) | Latency Ratio (SDPA / Structured) |
|---|---:|---:|---:|---:|---:|
| 1,024 | 0.11 ms | 12.2 MB | 0.15 ms | 50.2 MB | 0.73x (SDPA faster) |
| 2,048 | 0.42 ms | 22.3 MB | 0.20 ms | 66.3 MB | 2.10x |
| 4,096 | 1.65 ms | 42.4 MB | 0.40 ms | 98.4 MB | 4.13x |
| 8,192 | 8.09 ms | 82.6 MB | 0.85 ms | 162.7 MB | 9.52x |
| 16,384 | 28.62 ms | 163.1 MB | 1.81 ms | 291.2 MB | 15.81x |
| 32,768 | 108.15 ms | 324.1 MB | 3.65 ms | 548.2 MB | **29.63x** |

---

## 3. Analysis & Key Takeaways

1. **Quadratic vs. Linear Scaling:**
   - Standard SDPA exhibits clear quadratic scaling ($O(N^2)$), rising from 0.11 ms at 1k to 108.15 ms at 32k tokens.
   - The block-plus-low-rank structured layer scales linearly ($O(N)$), taking only 3.65 ms at 32k tokens.
2. **Trade-offs & Open Questions:**
   - **Baseline Overhead:** For short sequences ($N \le 1024$), FlashAttention kernel fusion beats the multi-step structured module.
   - **Memory Footprint:** The structured prototype creates intermediate un-fused PyTorch activation buffers (548 MB vs 324 MB at 32k), which can be reduced with a fused Triton/CUDA kernel.
   - **Quality / Perplexity:** This benchmark strictly proves computational complexity; measuring downstream perplexity on language modeling tasks remains the critical open evaluation gate.
