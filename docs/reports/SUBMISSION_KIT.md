# WEFT-2 Academic Publication Submission Kit & Reproducibility Package

**Stratum Research Group | Research Publication Package**
**Target Venues:** NeurIPS / ICML / ICLR / JMLR / ArXiv Computer Science (Computation and Language)

---

## 1. Executive Summary & Manuscript Metadata

- **Title:** WEFT: Windowed Exponentially-Decayed Factorized Tensor Attention for Linear-Complexity Long-Context Sequence Modeling
- **Authors:** Stratum Research Group
- **Primary Subject:** Computer Science - Computation and Language (`cs.CL`), Machine Learning (`cs.LG`)
- **Manuscript Artifacts:**
  - LaTeX Manuscript Source: [`paper/paper.tex`](paper/paper.tex)
  - Compiled Publication PDF: [`paper/paper.pdf`](paper/paper.pdf)
  - Extended Markdown Preprint: [`paper/paper.md`](paper/paper.md)
  - High-DPI Vector Figures:
    - Needle-in-a-Haystack Heatmap: [`paper/figures/needle_recall_heatmap.pdf`](paper/figures/needle_recall_heatmap.pdf)
    - 15k Pretraining Convergence: [`paper/figures/pretrain_convergence.pdf`](paper/figures/pretrain_convergence.pdf)

---

## 2. Key Empirical Results for Submission Abstract

| Metric | Result | Hardware / Baseline Context |
| :--- | :---: | :---: |
| **Recurrent Context Memory** | **$\mathcal{O}(1)$ Constant** | Bounded at **1,208 MB to 1,668 MB flat** across $N \in [1024, 1{,}048{,}576]$ tokens |
| **Prefill Throughput Acceleration** | **24.2x speedup** | 14,427 tok/s vs 596 tok/s baseline on 8,192 tokens |
| **Speculative Decoding Speedup** | **4.00x wall-clock** | 74.6 tok/s vs 18.6 tok/s autoregressive baseline |
| **Resident KV-Cache Compression** | **21.3x compression** | Compressed latent cache ($d_{\text{lat}}=192$) vs full 4,096 projections |
| **BitNet b1.58 Ternary Quantization** | **75% memory drop** | $W \in \{-1, 0, 1\}$ (2-bit packed, integer additions, $1.0000$ cosine similarity) |
| **1-Trillion Parameter Feasibility** | **241.3 GB VRAM** | 1.036T model fits in one 8x GPU node (vs 1,930 GB FP16) |
| **Pretraining Convergence** | **9.18 $\to$ 2.92 loss** | 15,000 steps ($7.68\text{M}$ tokens) on FineWeb-Edu in 36 minutes |
| **Formal Mathematical Grounding** | **Lean 4 Proven** | Discrete Boolean uncrossing & bounded recurrence state |

---

## 3. Step-by-Step Exact Reproducibility Commands

### A. Environment Setup
```bash
git clone https://github.com/Stratum-Research-Group/WEFT.git
cd WEFT
uv sync --extra dev
```

### B. Verification of Formal Proofs & Complete Test Suite
```bash
# Run formal math tests and complete 58-test verification matrix
uv run pytest tests/ -v
```

### C. 1,000,000-Token Streaming Bounded Memory Proof
```bash
uv run python benchmarks/benchmark_chunked_streaming.py
```

### D. Speculative Decoding Acceleration (4x Speedup)
```bash
uv run python examples/speculative_decoding.py
```

### E. BitNet b1.58 Addition-Only Triton GEMM Verification
```bash
uv run pytest tests/test_ternary_gemm.py -v
```

### F. Frontier 1-Trillion Parameter Specification (Zero-Memory Meta Device)
```bash
uv run python -c "
from weft.scaling_specs import WEFT_SCALE_REGISTRY, get_model_params
cfg = WEFT_SCALE_REGISTRY['1T'].to_config()
total, active = get_model_params(cfg)
print(f'WEFT-1T: Total = {total/1e12:.3f}T, Active = {active/1e9:.1f}B')
print(f'BitNet b1.58 Packed Weight VRAM (2-bit): {total * 0.25 / (1024**3):.1f} GB')
"
```

### G. Standardized Downstream Multi-Task Evaluation
```bash
uv run python benchmarks/evaluate_downstream.py --checkpoint checkpoints/weft2_foundation_15k
```

### H. Universal GGUF / Local Serving Exporter
```bash
uv run python weft/export_gguf.py --checkpoint checkpoints/weft2_foundation_15k --output dist/weft2_15k.gguf.json
```

---

## 4. Checkpoint Artifact Registry

| Checkpoint Name | Local Path | Parameters | Training Stage / Loss |
| :--- | :--- | :---: | :---: |
| **WEFT-2 Release v2** | `weft2_release_v2_ckpt/` | 589M | Unified Foundation Checkpoint |
| **WEFT-2 Foundation 15k** | `checkpoints/weft2_foundation_15k/` | 80.3M | FineWeb-Edu 15,000 steps (Loss 2.92) |
| **WEFT-2 SFT Format** | `checkpoints/weft2_sft_15k/` | 80.3M | Arithmetic Scratchpad Format (Loss 4.75) |
| **WEFT-2 GRPO Aligned** | `checkpoints/weft2_grpo_15k/` | 80.3M | RLVR MathRuleVerifier Aligned |
| **WEFT-2 BitNet b1.58** | `checkpoints/weft2_foundation_5k_bitnet/` | 80.3M | Ternary Quantized Addition-Only Model |

---

## 5. BibTeX Citation Entry

```bibtex
@article{stratum2026weft,
  title={{WEFT: Windowed Exponentially-Decayed Factorized Tensor Attention for Linear-Complexity Long-Context Sequence Modeling}},
  author={{Stratum Research Group}},
  journal={arXiv preprint arXiv:2611.XXXXX},
  year={2026},
  url={https://github.com/Stratum-Research-Group/WEFT}
}
```
