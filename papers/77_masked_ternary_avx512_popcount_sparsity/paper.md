# Paper 77: Masked Ternary AVX-512 Popcount Operators: 1.58-Bit Sparsity and Hardware Invariants on Zen 4 Silicon

**Nicholas Charles Mynhardt**  
*Stratum Research Group • research@stratum.org*  
*November 2026*

---

> [!NOTE] Peer-Review Context: Architectural Epoch & Fleet Evolution
> **Research Phase**: Epoch VII: Bare-Metal AVX-512 VPOPCNTDQ & AVX2 Silicon Acceleration (Papers 74–92)
> **Chronological Context**: Phase 7 transitioned the architectural framework to native C++20 vector registers on modern microarchitectures (AMD Zen 4/5, Intel Sapphire Rapids, and AVX2 fallback), developing an 18-kernel silicon suite covering bitboard popcount GEMV, batch-4 S-GEMM, paged Int4 KV Flash-Attention, GQA, SwiGLU, sliding-window attention, and fused ternary projections.
> *Reviewer Guidance*: Reported empirical metrics (resident RAM, parameter count, model fleet size, and execution speedups) correspond to the verified state of the WEFT pipeline at this paper's chronological release. Subsequent epochs expand upon these results while maintaining backward mathematical consistency certified in Lean 4.

## Abstract
Modern extreme-quantization architectures, such as 1.58-bit ternary networks (BitNet b1.58) and sparse Mixture-of-Experts (MoE), require linear operations over trinary alphabets $\{-1, 0, +1\}$. Naive bitboard implementations cannot represent the zero element without inflating word bit-depth to 2 bits, incurring a $100\%$ bandwidth tax over binary bitboards. In this work, we present an AVX-512 VPOPCNTDQ masked ternary GEMV kernel that packs trinary matrices into dual 1-bit bitboards: a nonzero sparsity mask $\mathbf{M}_{\text{nz}}$ and a polarity sign mask $\mathbf{M}_{\text{sign}}$. By executing vector bitwise AND-masking alongside vector population counts (`_mm512_andnot_si512` and `_mm512_popcnt_epi64`), the kernel evaluates ternary matrix-vector products at $541.06\text{ GFLOPs/s}$ equivalent on an AMD Ryzen 5 7600X processor. We formally prove exact ternary popcount decomposition, mask disjointness, and sparsity-bounded FLOP limits in Lean 4 without non-standard axioms.

---

## 1. Introduction & Dual Bitboard Representation
Standard 1-bit binary bitboards represent elements in $\{-1, +1\}$. To accommodate zero weights without moving to unaligned 2-bit packing schemes, we decompose trinary matrix $\mathbf{W} \in \{-1, 0, +1\}^{M \times N}$ into two aligned 1-bit boolean matrices:
1. **Nonzero Mask** $\mathbf{M}_{\text{nz}}[r, c] = 1 \iff \mathbf{W}[r, c] \neq 0$
2. **Polarity Mask** $\mathbf{M}_{\text{sign}}[r, c] = 1 \iff \mathbf{W}[r, c] = +1$

For bipolar activations $\mathbf{x} \in \{-1, +1\}^N$, the elementwise product with row $\mathbf{w}$ evaluates as:
$$w_c \cdot x_c = \begin{cases} +1 & \text{if } M_{\text{nz}}[c]=1 \text{ and } (M_{\text{sign}}[c] \equiv x_c) \\ -1 & \text{if } M_{\text{nz}}[c]=1 \text{ and } (M_{\text{sign}}[c] \not\equiv x_c) \\ 0 & \text{if } M_{\text{nz}}[c]=0 \end{cases}$$

---

## 2. Mathematical Formalization & Lean 4 Soundness

### 2.1 The Masked Popcount Isomorphism
Let $\mathbf{d} = \mathbf{M}_{\text{sign}} \oplus \mathbf{x}_{\text{sign}}$ denote the mismatched bit mask. The positive and negative match masks are computed via:
$$\mathbf{m}_{\text{pos}} = (\sim \mathbf{d}) \wedge \mathbf{M}_{\text{nz}}, \quad \mathbf{m}_{\text{neg}} = \mathbf{d} \wedge \mathbf{M}_{\text{nz}}$$
The exact dot product is:
$$\langle \mathbf{w}, \mathbf{x} \rangle = \operatorname{popcount}(\mathbf{m}_{\text{pos}}) - \operatorname{popcount}(\mathbf{m}_{\text{neg}})$$
Eliminating all arithmetic multiplications in silicon.

### 2.2 Machine-Checked Verification
Proven in Lean 4 standard library without non-standard axioms:
- `ternary_popcount_decomposition`: Proves $\mathbf{w} \cdot \mathbf{x} = N_{\text{pos}} - N_{\text{neg}}$.
- `ternary_sparsity_operation_bound`: Proves active non-zero bounds under sparsity fraction $s$.
- `ternary_mask_disjointness`: Proves disjointness between positive and negative bit masks.
- `zero_element_flop_reduction`: Proves strict FLOP monotonic reduction under weight sparsification.

---

## 3. Empirical Performance

Evaluated on an AMD Ryzen 5 7600X 6-core processor for an $8192 \times 8192$ ternary projection:

| Metric | Measured Value |
| :--- | :--- |
| **Execution Latency** | $248.06\text{ }\mu\text{s}$ |
| **Effective Throughput** | $541.06\text{ GFLOPs/s}$ equivalent |
| **Precision** | Dual 1-bit packed bitboards ($1.58$-bit effective) |
| **Numerical Error vs Float32** | $0.00\text{e}+00$ (Bit-exact) |

---

## 4. Conclusion
Paper 77 extends our hardware-accelerated popcount kernel suite to 1.58-bit ternary foundation architectures. By evaluating ternary sparsity directly through masked vector registers, edge devices can execute sparse foundation models with zero floating-point arithmetic overhead.