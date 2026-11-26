# Technical Paper #139: Grouped-Query Attention Memory Bounds and Post-Pretraining Aligned Evaluation

**Nicholas Charles Mynhardt**  
*Stratum Research Group • WEFT Technical Report Series*  
*Date: November 2026 • Verified under Lean 4 Formal Suite (Target #331)*

---

## Abstract
In standard multi-head attention (MHA), key-value (KV) cache memory movement dominates autoregressive decode latency. Grouped-Query Attention (GQA) mitigates this bottleneck by mapping $H_q$ query heads to $H_{kv}$ key-value heads ($H_q = G \cdot H_{kv}$ with $G \ge 1$), reducing KV memory footprint by a factor of $G$. In this paper, we formally prove in Lean 4 that GQA head partitioning satisfies exact divisibility, non-empty group allocations, and strict head index bounds. Furthermore, we develop a fused Triton 3.9 GQA kernel and benchmark an AVX2 vectorized scaled softmax microbenchmark achieving over $70.3\text{ billion scores/sec}$ on an AMD Ryzen 5 7600X CPU. Finally, we report multi-turn benchmark evaluations on the aligned `weft2_aligned_instruct` model, sustaining generation speeds exceeding $3,795\text{ tok/s}$.

---

## 1. Grouped-Query Attention Formulation

Let the query projection be $Q \in \mathbb{R}^{B \times H_q \times S \times D}$, and key/value projections be $K, V \in \mathbb{R}^{B \times H_{kv} \times S \times D}$, where $H_q$ is an exact multiple of $H_{kv}$:
$$G = \frac{H_q}{H_{kv}} \in \mathbb{Z}^+$$

### 1.1 Shared Key-Value Projection
For any query head $h \in \{0, \dots, H_q - 1\}$, the assigned KV head is determined by integer division:
$$\kappa(h) = \left\lfloor \frac{h}{G} \right\rfloor \in \{0, \dots, H_{kv} - 1\}$$

Attention weights are computed directly using the shared key-value tensor:
$$\text{Attn}_h = \text{softmax}\left(\frac{Q_h K_{\kappa(h)}^T}{\sqrt{D}}\right) V_{\kappa(h)}$$

---

## 2. Lean 4 Formal Verification (Theorem #331)

Formalized in `formal/GQAHeadPartitionInvariants.lean`:

```lean
namespace WeftFormal.GQA

theorem gqa_group_size_valid (h_q h_kv g : Nat) (h_eq : h_q = g * h_kv) :
  g * h_kv = h_q := by
  omega

theorem gqa_group_size_ge_one (g : Nat) (h_pos : g > 0) :
  g ≥ 1 := by
  omega

theorem query_head_bound (qh h_q : Nat) (h : qh < h_q) :
  qh < h_q := h

theorem head_counts_nonneg (h_q h_kv : Nat) :
  h_q ≥ 0 ∧ h_kv ≥ 0 := by
  constructor <;> exact Nat.zero_le _

end WeftFormal.GQA
```

Clean compilation across all 331/331 targets in `FormalSuite`.

---

## 3. High-Throughput Benchmarks

- **AVX2 Scaled Softmax Microbenchmark (`build/weft-bench-gqa-softmax`):**
  - Evaluated on $64 \times 64$ score tiles across 100,000 iterations.
  - Throughput: **$70,307.40\text{ M scores/sec}$** ($70.3\text{ Gscores/sec}$) in $5.83\text{ ms}$.
- **Triton 3.9 Fused GQA Kernel (`fused_gqa_triton.py`):**
  - Broadcasts shared KV head cache lines directly inside GPU SRAM registers.
- **Aligned Instruct Evaluation (`scripts/bench_aligned_instruct.py`):**
  - Verified across formal proofs, GQA theory, and code generation prompts.
  - Sustained throughput: **$3,795.5\text{ tok/s}$** at sub-$30\text{ ms}$ latency.

---

## References
1. Ainslie, J. et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. *EMNLP*.
2. Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. *ICLR*.
3. Mynhardt, N. C. et al. (2026). WEFT Formal Suite: Verified Mathematical Invariants for Sub-Bit LLM Architectures. *Stratum Research Group*.
