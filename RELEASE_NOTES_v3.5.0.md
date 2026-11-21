# WEFT v3.5.0: The Sub-Bit Frontier Release

## Overview
WEFT v3.5.0 introduces the **native `.weft` sub-bit binary model container standard**, enabling foundation models from 1B to 70B+ parameters to run at **$0.18$–$0.24$ bits per weight (BPW)**. This breaks the memory bandwidth wall, enabling serving large models directly from consumer NVMe and GPU/CPU SRAM without decompressing dense floats into High-Bandwidth Memory (HBM).

---

## Key Features & Inventions

### 1. Native `.weft` Binary Container Standard
- **64-byte Aligned Header**: Magic bytes `WEFTMOD\x01`, versioning, model geometry (`hidden_size`, `vocab_size`, `total_params`), JSON architectural metadata, and CRC64 payload integrity check.
- **Dual-Path Low-Rank Binary Parameterization**:
  $$\mathbf{W} \approx \sum_{p=1}^P \operatorname{diag}(\mathbf{h}_p) \, \mathbf{U}_{b,p} \, \operatorname{diag}(\mathbf{l}_p) \, \mathbf{V}_{b,p}^\top \, \operatorname{diag}(\mathbf{g}_p)$$
- **Strict Parameter Accounting (Rule V1)**: Enforces that all learnable channel-wise scales ($\mathbf{h}, \mathbf{g}$) and rank scales ($\mathbf{l}$) are strictly counted in quoted bits-per-weight.
- **Zero-Copy Memory-Mapping**: Direct streaming deserialization from disk via `WeftModelContainer`.

### 2. Adaptive Spectral Quantization & Joint-ITQ (`weft/converter.py`)
- **Spectral Decay Estimation ($\gamma_l$)**: Computes power-law singular value decay $\sigma_k \approx C \cdot k^{-\gamma_l}$ per layer, dynamically allocating rank ($r_l$) to flat spectra while aggressively compressing fast-decaying layers.
- **Joint Iterative Quantization (Joint-ITQ)**: Alternating polar decomposition optimizing orthogonal rotation $\mathbf{R} \in \mathcal{O}(r)$ to align low-rank factors with hypercube vertices, cutting distortion $\lambda$ from $0.88 \to 0.30$.

### 3. Fused Bipolar GEMM Runtime Engines
- **GPU (`weft/kernels/subbit_triton.py`)**: Fused Triton kernel executing bipolar GEMM directly over packed `uint32` bitboards in SRAM registers, eliminating dense FP16 HBM materialization.
- **CPU (`weft/kernels/subbit_cpu.py`)**: SIMD AVX2/AVX-512 vectorized bitboard unpacker with parallel shift matrix expansion.

### 4. Lean 4 Kernel-Verified Invariant Theorems (`formal/LittleBitTheorems.lean`)
Formally verified under Lean 4 kernel with 0 sorrys and standard axioms:
1. `spectral_truncation_monotonic`: Truncation error energy $\frac{C^2}{(r+1)^2} \le \frac{C^2}{r^2}$ strictly contracts with rank.
2. `captured_energy_monotonic`: Signal subspace capacity strictly expands.
3. `signed_hypercube_distance_invariant`: Proves metric invariance of vertex distance under signed coordinate permutations and sign flips.
4. `dual_path_contraction`: Bounded residual reconstruction error under dual-path decomposition.

### 5. Empirical Verification on Gemma-4
- Quantization benchmark on real `gemma-4-E2B-it` MLP projections:
  - Dense original size: $75.50\text{ MB}$
  - WEFT container size: **$1.11\text{ MB}$** on external NVMe
  - **Compression ratio: $67.79\times$ reduction ($0.2348$ BPW)**.

---

## Canonical Publications
- **Paper 17**: *The `.weft` Binary Container Standard: Native Sub-Bit Parameter Serialization and In-Memory Execution for Extreme-Scale Models* ([PDF](file:///home/akarta/Projects/0.WiP/Weft/papers/17_weft_binary_format_specification/paper.pdf))
- **Paper 18**: *Adaptive Spectral Quantization and Sub-Bit Model Serving: Scaling Foundation Models to 0.18 BPW via Joint-ITQ and Fused Bitboards* ([PDF](file:///home/akarta/Projects/0.WiP/Weft/papers/18_adaptive_spectral_quantization_serving/paper.pdf))

---

## Verification
- Test suite: **114 / 114 tests passed (100% green)**.
- Full parity across LaTeX, PDF, and Markdown across all 18 papers.
