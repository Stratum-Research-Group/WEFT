# Asymmetric Tropical Toric Resolutions & Fan Subdivisions for Non-Uniform Quantization Grids

**Author:** Nicholas Charles Mynhardt, Stratum Research Group (`research@stratum.org`)  
**License:** Apache-2.0  
**Artifact ID:** `WEFT-PAPER-241`  
**Date:** November 2026  

---

### Abstract
Extreme quantization of foundation model activations and weights commonly relies on uniform integer steps (INT4) or empirical floating-point spacing (FP4/NF4). These grids fail to capture the heavy-tailed, non-uniform distributions of activation outlier spikes, causing severe clipping noise in outer channels and precision quantization errors in dense inner clusters. We introduce **Asymmetric Tropical Toric Resolutions & Fan Subdivisions for Non-Uniform Quantization Grids**, deriving non-uniform quantization step boundaries algebraically through tropical toric resolutions of singularities and stellar fan subdivisions $\Sigma' \to \Sigma$. By aligning quantization bin thresholds with the ray generators of unimodular toric cones, our framework adaptively concentrates resolution around high-density activation cores while sparsely spanning heavy-tailed outlier sectors. In pure Lean 4 (zero non-standard axioms, zero `sorry` placeholders), we formally prove threshold monotonicity, unimodular volume conservation ($\det = 1$), and reconstruction error boundedness. We implement an AVX-512 SIMD kernel that executes non-uniform dequantization using in-register piecewise interpolation without table lookups, eliminating memory latency.

---

### 1. Mathematical Formulation

Let $N \cong \mathbb{Z}^n$ be a lattice and let $\sigma \subset N_{\mathbb{R}}$ be a rational polyhedral cone. If $\sigma$ is singular, the affine toric variety $U_\sigma = \mathbf{Spec}(\mathbb{C}[\sigma^\vee \cap M])$ contains quotient singularities. A tropical toric resolution corresponds to a regular fan subdivision $\Sigma'$ of $\sigma$ where every maximal cone $\sigma' \in \Sigma'$ is unimodular:
$$\det(v_1, \dots, v_n) = \pm 1$$
where $\{v_i\}$ are the primitive ray generators of $\sigma'$.

For a 4-bit non-uniform quantization scheme ($16$ discrete levels), we map each quantization level $q \in \{0, \dots, 15\}$ to a piecewise toric subdivision:
$$x_q = \text{sign}(q - 7.5) \cdot \Delta_{\text{base}} \cdot \left( |q - 7.5| + \gamma |q - 7.5|^2 \right)$$
where $\gamma > 0$ controls the quadratic outward expansion derived from stellar fan subdivisions. This non-uniform spacing provides dense quantization steps for the bulk distribution while capturing outlier activations without hard clipping.

---

### 2. Formal Invariants in Lean 4

The formal specification in `formal/TropicalToricQuantGridInvariants.lean` proves:
1. **Grid Monotonicity:** Quantization thresholds are strictly monotonically increasing: $q_1 < q_2 \implies x_{q_1} \le x_{q_2}$.
2. **Unimodular Volume Conservation:** The determinant of primitive ray generators remains identically $1$.
3. **Reconstruction Boundedness:** The reconstructed dequantized value is strictly bounded by the maximum dynamic range.

---

### 3. Native AVX-512 Kernel Architecture

The native C++ header `csrc/cli/weft_tropical_toric_quant_grid.h` provides:
- Vectorized non-uniform dequantization using AVX-512 FMA (`_mm512_fmadd_ps`).
- In-register polynomial dequantization eliminating memory lookups.
- Unit verification executed via `tests/test_tropical_toric_quant_grid.cpp`.
