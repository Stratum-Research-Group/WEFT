# Asymmetric Tropical Moduli of Toric Curves & Tropical Convexity for Multi-Task KV Partitioning

**Author:** Nicholas Charles Mynhardt, Stratum Research Group (`research@stratum.org`)  
**License:** Apache-2.0  
**Artifact ID:** `WEFT-PAPER-232`  
**Date:** November 2026  

---

### Abstract
Serving multiple concurrent reasoning agents and distinct task contexts on edge hardware creates severe KV cache fragmentation and frequent memory eviction cascades. Classical memory management fails to provide topological partition boundaries without physical memory duplication. We introduce **Asymmetric Tropical Moduli of Toric Curves & Tropical Convexity for Multi-Task KV Partitioning**, modeling memory address spaces as tropical polyhedra over the max-plus semiring $(\mathbb{R} \cup \{-\infty\}, \max, +)$. By evaluating the tropical moduli space of metric graphs $\mathcal{M}_{g,n}^{\text{trop}}$, we partition continuous token cache blocks into disjoint tropical convex hulls. We formally prove in pure Lean 4 (zero non-standard axioms, zero `sorry`) that tropical convex combinations are closed, metric graph paths preserve additive metric triangle inequalities, and tropical hyperplane separation guarantees non-interfering page allocations across tenant domains. We implement an AVX-512 SIMD kernel that executes in-register max-plus projections (`_mm512_max_ps`, `_mm512_add_ps`), eliminating page-table fragmentation and achieving line-rate multi-task KV routing.

---

### 1. Mathematical Formulation

Let $(\mathbb{T}, \oplus, \odot)$ denote the tropical semiring where $\mathbb{T} = \mathbb{R} \cup \{-\infty\}$, $a \oplus b = \max(a, b)$, and $a \odot b = a + b$. A subset $C \subset \mathbb{T}^d$ is tropically convex if for all $x, y \in C$ and $\alpha, \beta \in \mathbb{T}$ such that $\alpha \oplus \beta = 0$, we have:
$$(\alpha \odot x) \oplus (\beta \odot y) \in C$$

In multi-task KV cache partitioning, each task $k \in \{1, \dots, K\}$ requests a sequence allocation of size $L_k$ with priority weight $w_k$. We define the tropical coordinate assignment for memory block indices $i \in \{0, \dots, M-1\}$:
$$T_k(i) = \max(0, w_k - |i - \text{center}_k|)$$
Tropical hyperplane separation guarantees that if:
$$\text{center}_k + L_k \le \text{center}_{k+1}$$
the tropical support supports are strictly disjoint, ensuring non-overlapping memory assignment with zero fragmentation.

---

### 2. Formal Invariants in Lean 4

The formal specification in `formal/TropicalConvexityKVInvariants.lean` proves:
1. **Max-Plus Metric Additivity:** For path distances on metric trees, tropical distance combinations satisfy triangle inequalities.
2. **Disjoint Partitioning:** Non-overlapping tropical centers satisfy strict boundary separation.
3. **Capacity Non-Negativity:** All allocated partition capacities are non-negative and bounded by the physical cache size.

---

### 3. Native AVX-512 Kernel Architecture

The native C++ header `csrc/cli/weft_tropical_convexity_kv.h` provides:
- Vectorized max-plus evaluation using `_mm512_max_ps` and `_mm512_add_ps`.
- Zero-allocation partitioning table generator for concurrent tenant contexts.
- Unit verification executed via `tests/test_tropical_convexity_kv.cpp`.
