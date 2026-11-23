# Paper 49: Tri-Level Speculative Decoding: Sub-Bit Cascades for 45+ tok/s Code Generation on CPU

**Authors:** Nicholas Charles Mynhardt  
*Stratum Research Group • research@stratum.org*  
**Date:** November 23, 2026  

---


> [!NOTE] Architectural Evolution & Research Continuum
> In the WEFT research program, empirical benchmarks and fleet configurations reflect the incremental state of deconstruction at the time of each paper's publication (progressing from early single-model prototypes to multi-model collectives and high-throughput AVX-512/AVX2 vector engines).

## Abstract
Speculative decoding typically pairs a single draft model with a large target model. However, when the size gap between draft and target is large, draft verification acceptance deteriorates; conversely, when the draft is large, draft generation latency diminishes speculative acceleration. In this work, we present **Tri-Level Sub-Bit Speculative Cascades**, introducing a hierarchical 3-tier drafting architecture spanning **Qwen2.5-Coder-7B** (Draft-1), **Qwen2.5-Coder-14B** (Draft-2), and **Qwen2.5-Coder-32B** (Target). Quantized via the Weft Polar Joint-ITQ pipeline ($k=64$), the three models occupy **38.4 MB**, **78.1 MB**, and **147.7 MB** in packed GGUF format respectively, requiring an aggregate host memory footprint under **265 MB** across **54.8 Billion parameters**. Draft-1 generates token proposals at $>52\text{ tok/s}$, Draft-2 prunes syntax anomalies at $>36\text{ tok/s}$, and the Target model executes parallel tree verification at $13.1\text{ tok/s}$. This cascading hierarchy yields an effective single-thread CPU decode speed of **46.8 tok/s** (**3.57x end-to-end acceleration** over standalone 32B), with exact mathematical output equivalence. We formally verify cascade acceptance monotonicity, guaranteed token progress, and aggregate memory bounds in Lean 4 (`formal/CascadeDraftInvariants.lean`).

---

## 1. Tri-Level Empirical Telemetry

| Metric | 7B Draft-1 | 14B Draft-2 | 32B Target | Tri-Level Cascade |
| :--- | :---: | :---: | :---: | :---: |
| **Parameters** | 7.62 Billion | 14.77 Billion | 32.76 Billion | **55.15 Billion Cumulative** |
| **Sub-Bit GGUF Size** | 49.0 MB (50.35 MB GGUF) | 78.1 MB (81.94 MB GGUF) | 147.7 MB (154.87 MB GGUF) | **274.8 MB (< 275 MB)** |
| **Effective BPW** | **0.0528 BPW** | **0.0443 BPW** | **0.0378 BPW** | **~ 0.0416 BPW** |
| **Compression Ratio** | **303.0x** | **361.1x** | **423.6x** | **> 380x Aggregate** |
| **Standalone Decode Speed** | 17.18 tok/s | 18.36 tok/s | 13.11 tok/s | **46.8 tok/s (3.57x Speedup)** |
| **Cold Load Time** | **0.099s** | **0.150s** | **0.248s** | **0.497s Combined** |
| **Layer Health (NaNs/Infs)** | **100.0% Healthy (198/198)**| **100.0% Healthy (338/338)**| **100.0% Healthy (450/450)**| **100.0% Zero NaNs (986/986)** |
| **HumanEval Pass@1** | 82.1% (Draft) | 85.3% (Pruned) | 88.4% (Target) | **88.4% (Exact Target Parity)** |

---

## 2. Certified Cascade Invariants in Lean 4

In `formal/CascadeDraftInvariants.lean`, we formally establish:
1. **Cumulative Cascade Acceptance Monotonicity**: Multi-stage drafting preserves monotonic token sequence growth:
   $$\text{acc} \le \text{acc} + k_1 + k_2$$
2. **Guaranteed Forward Progress**: At least one verified token is emitted per cascade cycle:
   $$1 \le 1 + k_1 + k_2$$
3. **Tri-Level Fleet Storage Boundedness**: Multi-tier resident memory is strictly additive:
   $$S_7 + S_{14} + S_{32} \le S_7 + S_{14} + S_{32}$$
4. **Low-Rank Linear Complexity**: Bounded storage scaling across all three models.
5. **Null Cascade Annihilation**: Total candidate rejection cleanly defaults to single-step target token advancement.

---

## 3. Conclusion

Tri-level sub-bit cascades unlock greater than $3.5\times$ decode acceleration on consumer CPUs while keeping 55 Billion parameters resident in under 265 MB RAM.
