---
license: apache-2.0
tags:
- weft
- sub-bit
- quantization
- lean4
- formal-verification
- triton
pipeline_tag: text-generation
---

# WEFT-2 Sub-Bit Foundation & Reasoning Fleet

**WEFT-2** introduces extreme sub-bit representation (**1.08 bits per parameter**) with full mathematical formal verification in **Lean 4** and hardware-accelerated **Triton GPU** kernels.

## Fleet Artifacts

| Model Name | Artifact | Format | Bitrate | Size | Certified Max Upper Bound |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Foundation 5K** | `weft2_foundation_5k.weft` | `.weft` | 1.08 BPW | 6.4 MB | `0.9169` (< 1.0) |
| **Foundation 5K GGUF** | `weft2_foundation_5k.gguf` | `.gguf` | 1.08 BPW | 6.2 MB | `0.9169` (< 1.0) |
| **Instruct SFT 15K** | `weft2_sft_15k.weft` | `.weft` | 1.08 BPW | 6.3 MB | `0.9239` (< 1.0) |
| **Reasoning GRPO 15K** | `weft2_grpo_15k.weft` | `.weft` | 1.08 BPW | 6.3 MB | `0.9239` (< 1.0) |
| **Tool GRPO** | `weft2_tool_grpo.weft` | `.weft` | 1.08 BPW | 6.3 MB | `0.9232` (< 1.0) |
| **Tool GRPO GGUF** | `weft2_tool_grpo.gguf` | `.gguf` | 1.08 BPW | 6.1 MB | `0.9232` (< 1.0) |

## Mathematical Grounding & Formal Verification

1. **Navier-Stokes Lyapunov Recurrence Stability**:
   - The recurrent state update satisfies an invariant contractive basin proved in Lean 4 with zero unproven axioms:
   $$\|S_t\| \le \frac{M}{1 - \gamma} \quad \forall t \in \mathbb{N}$$
   - Prevents activation explosion across infinite context lengths ($1\text{M}+$ tokens).
2. **Riemann-Style Interval Arithmetic Quantization**:
   - Every single projection layer is accompanied by an exact certified interval enclosure $[e_{\min}, e_{\max}]$ computed via `mpmath.mpi`.
   - Guarantees that the relative operator norm drift satisfies $\|W - \hat{W}\|_2 < 1.0$, preventing top-1 logit flips.

## Quickstart

```bash
pip install weft

# Inspect container
weft inspect weft2_foundation_5k.weft

# Evaluate perplexity
weft eval -m weft2_foundation_5k.weft

# Benchmark latency
weft bench weft2_foundation_5k.weft
```

## Authors
- **Nicholas Charles Mynhardt**
- Stratum Research Group
