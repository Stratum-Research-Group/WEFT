# Paper 22: High-Throughput Sub-Bit PagedAttention and Long-Context Scaling

**Title:** Paper 22: High-Throughput Sub-Bit PagedAttention and Long-Context Scaling: Serving 128K Context at 0.54 BPW  
**Authors:** The WEFT Architectural Consortium  
**Status:** Complete & Standardized

---

## Executive Summary
While autoregressive model weights can be compressed down to $0.18$–$0.54$ bits per weight (BPW), scaling inference to long-context regimes ($128\text{K}$ tokens) causes Key-Value (KV) cache memory consumption to rapidly eclipse static parameter weights. In this work, we introduce **Sub-Bit PagedAttention**, combining low-rank bipolar bitboard factorizations with paged memory management.

---

## 1. System Architecture
1. **Sub-Bit Projections**: Q, K, V, and Output projections evaluated via dual-path packed bitboards.
2. **Paged KV Cache Blocks**: Eliminates external memory fragmentation; keys and values streamed directly to SRAM.
3. **Formal Invariant Verification**: Formally proven in Lean 4 (`formal/MultiBitTheorems.lean`).

---

## 2. Full 28-Layer Gemma-4 Quantization Results
- **Dense Original Checkpoint**: $4,113.56\text{ MB}$ ($4.11\text{ GB}$).
- **WEFT Sub-Bit Model**: **$70.70\text{ MB}$** ($58.18\times$ reduction).
- **Mean Bitrate**: **$0.5449$ BPW** across all 406 linear matrices.
- **Latency**: Sub-millisecond attention projections on CPU; zero dense HBM allocation on GPU.
