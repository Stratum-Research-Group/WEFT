# Asymmetric Tropical Moduli of Weighted Graphs & Tropical Splitting for Dynamic KV Cache Eviction

**Author:** Nicholas Charles Mynhardt, Stratum Research Group (`research@stratum.org`)  
**License:** Apache-2.0  
**Artifact ID:** `WEFT-PAPER-250`  
**Date:** November 2026  

---

### Abstract
In long-context retrieval-augmented generation (RAG) and multi-document reasoning tasks, key-value caches become saturated with irrelevant context from completed document spans. Conventional eviction policies prune tokens uniformly across documents, fragmenting active conversational context and causing document cross-contamination. We present **Asymmetric Tropical Moduli of Weighted Graphs & Tropical Splitting for Dynamic KV Cache Eviction**, formalizing multi-document conversational histories as metric weighted graphs $(\Gamma, \ell, w)$ in the tropical moduli space $\mathcal{M}_{g,n}^{\text{trop}}$. By computing tropical vertex splittings along metric bridge edges separating distinct document clusters, our framework isolates completed document subgraphs from active query paths. In pure Lean 4 (zero non-standard axioms, zero `sorry` placeholders), we formally prove bridge edge separability, metric weight additivity across connected components, and bounded memory reclamation. We implement an AVX-512 SIMD kernel that executes vectorized cluster thresholding and in-register page deallocation without memory compaction latency.

---

### 1. Mathematical Formulation

Let $\mathcal{M}_{g,n}^{\text{trop}}$ be the tropical moduli space parameterizing tropical curves of genus $g$ with $n$ marked points, realized as metric ribbon graphs $\Gamma = (V, E, \ell)$. The length function $\ell: E \to \mathbb{R}_{>0}$ assigns edge weights corresponding to token-to-token semantic distances.

A bridge edge $e \in E$ is a tropical bottleneck if $\Gamma \setminus \{e\} = \Gamma_1 \sqcup \Gamma_2$. For multi-document KV caches, bridge edges separate the active dialogue component $\Gamma_{\text{active}}$ from document components $\Gamma_{\text{doc}, k}$. Tropical vertex splitting allows the eviction engine to deallocate entire subgraphs:
$$\text{Evict}(\Gamma_{\text{doc}, k}) \iff \text{dist}_{\Gamma}(\Gamma_{\text{doc}, k}, \Gamma_{\text{active}}) > \tau_{\text{bridge}}$$
Because eviction occurs at tropical bridge boundaries, topological graph connectivity of the active query is strictly preserved with zero fragment spill.

---

### 2. Formal Invariants in Lean 4

The formal specification in `formal/TropicalWeightedGraphKVInvariants.lean` proves:
1. **Bridge Metric Additivity:** For path traversal across a bridge edge, total distance is the sum of component distances plus bridge length.
2. **Component Independence:** Pruning an inactive component leaves active component indices unmodified.
3. **Reclaimed Capacity Boundedness:** Reclaimed page count is non-negative and bounded by initial allocation.

---

### 3. Native AVX-512 Kernel Architecture

The native C++ header `csrc/cli/weft_tropical_weighted_graph_kv.h` provides:
- Vectorized cluster distance evaluation using AVX-512 comparisons (`_mm512_cmp_ps_mask`).
- In-register cluster eviction flag generation without heap reallocation.
- Unit verification executed via `tests/test_tropical_weighted_graph_kv.cpp`.
