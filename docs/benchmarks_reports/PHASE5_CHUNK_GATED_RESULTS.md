# Phase 5 Architecture: Fused Chunk-Recurrent Gated Attention

**Hardware:** NVIDIA GeForce RTX 5060 Ti (16 GB VRAM)  
**Implementation:** [fused_chunk_gated.py](../benchmarks/fused_chunk_gated.py)

---

## 1. Architectural Innovation: Constant-Size Inter-Chunk State

In Phase 4, un-fused PyTorch code materialized all intermediate time steps across the sequence, consuming gigabytes of memory.

In **Phase 5**, we designed a **Chunk-Recurrent Gated Architecture** (similar in spirit to RetNet and Lightning Attention):
- Sequences are partitioned into chunks of size $C = 256$.
- **Intra-Chunk Attention:** Computed in parallel using local FlashAttention blocks ($O(C)$ memory).
- **Inter-Chunk Global Context:** Transmitted via a running state matrix $S$ of size $(\text{Batch}, \text{Rank}, D_{\text{model}})$.
- **State Footprint:** For $d_{\text{model}} = 1024, \text{rank} = 32$, the recurrent state size is strictly:
  $$1 \times 32 \times 1024 \times 2 \text{ bytes} = \mathbf{65.5 \text{ Kilobytes!}}$$
- **Decay Gating Across Chunks:**
  $$S_{k} = \gamma^C \odot S_{k-1} + \sum_{j \in \text{chunk}_k} \gamma^{C - 1 - j} (k_j^T x_j)$$

---

## 2. Empirical Results on RTX 5060 Ti

```
Config: d_model=1024, n_heads=16, batch_size=1, chunk_size=256, rank=32, fp16
```

| Sequence Length ($N$) | Standard Causal SDPA (PyTorch) | Chunk-Gated Structured Attention | Measured Speedup | Peak VRAM |
|---:|---:|---:|---:|---:|
| **1,024** | 7.93 ms (warmup) | 308.89 ms (warmup) | — | 66.4 MB |
| **2,048** | 0.31 ms | 1.47 ms | 0.2x | 92.6 MB |
| **4,096** | 0.83 ms | 1.78 ms | 0.5x | 144.8 MB |
| **8,192** | 3.26 ms | 3.56 ms | 0.9x | 249.3 MB |
| **16,384** | 12.28 ms | **6.47 ms** | **1.9x** | 458.3 MB |
| **32,768** | 47.82 ms | **13.05 ms** | **3.7x** | 876.3 MB |
| **65,536** | 192.31 ms | **28.54 ms** | **6.7x** | 1,712.3 MB |
| **131,072** | 769.94 ms | **58.81 ms** | **13.1x** | 3,384.3 MB |
| **262,144** | 3,130.00 ms (3.13 s) | **128.44 ms** | **24.4x** | 6,728.3 MB |

---

## 3. Analysis & Key Takeaways

1. **Strictly Linear $O(N)$ Scaling Without Memory Explosions:**
   - From 16k to 262k tokens, runtime scaled linearly from **6.47 ms** to **128.44 ms** ($2\times$ per sequence doubling).
   - In standard FlashAttention, runtime exploded from **12.28 ms** to **3,130.00 ms** (3.13 seconds, $4\times$ per sequence doubling).
2. **Break-Even Point:**
   - Crosses standard FlashAttention speed at **$N \approx 9{,}000$ tokens**.
   - At 262k tokens, it is **24.4x faster** than FlashAttention on a consumer GPU.
3. **State Compactness:**
   - The entire historical context across hundreds of thousands of tokens is compressed into a **65 KB** recurrent state matrix $S$, eliminating the massive KV-cache memory wall during autoregressive generation.
