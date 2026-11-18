# Phase 3 Empirical Gate: Real-World Language Modeling

**Task:** Next-token autoregressive character-level language modeling on TinyShakespeare  
**Hardware:** NVIDIA GeForce RTX 5060 Ti  
**Training Steps:** 1,000 steps per model  
**Implementation:** [benchmark_lm_training.py](file:///home/akarta/Projects/0.WiP/Weft/benchmark_lm_training.py)

---

## 1. Head-to-Head Empirical Metrics

| Metric | Standard Causal Transformer (Softmax) | Causal Structured Attention (Window 64 + Rank 32) | Delta / Finding |
|---|---:|---:|---|
| **Validation Loss** | **1.6249** | 2.8336 | +1.21 higher loss |
| **Validation Perplexity** | **5.08** | 17.01 | **3.35x worse perplexity** |
| **Training Time (1000 steps)** | **21.13 s** | 74.80 s | 3.5x slower training |
| **Step Latency** | **21.13 ms** | 74.80 ms | Un-fused PyTorch overhead |
| **Peak VRAM** | **627.6 MB** | 2,349.9 MB | Cumulative state expansion |

---

## 2. Text Generation Output Comparison

### Standard Transformer (Loss: 1.62, PPL: 5.08)
> *"ROMEO:*  
> *Why sweet many less the housand of his, bid, with my life,*  
> *Titure's door thy breath'd made a maid."*  
*(Coherent English words, proper character naming, grammatical syntax).*

### Structured Attention (Loss: 2.83, PPL: 17.01)
> *"ROMEO:*  
> * hot*  
> *hrdSois ne, sehoem be. se chn cyyon*  
> *Ganit*  
> *IB B:*  
> *Wa*  
> *osr a diyerw!NT*  
> *Yabrlavee y ne msson e, mhit"*  
*(Jumbled non-words, struggling with basic bigram/trigram consistency).*

---

## 3. The Root Cause: Why Pure Linear Cumulative Outer-Products Fall Behind

1. **The Inability to "Forget" / Decay in Simple Cumsums:**
   In standard attention, $\text{softmax}(QK^T)$ dynamically sharpens onto the few most relevant tokens and normalizes them cleanly.  
   In our linear accumulator:
   $$S_t = \sum_{\tau=1}^t k_\tau^T x_\tau$$
   All past tokens are simply summed with equal weight over time. Tokens from 200 steps ago drown out the immediate grammatical context, creating severe gradient interference and plateauing at a high loss (~2.83).
2. **Missing Input-Dependent Gating:**
   Modern linear attention architectures that work in production (e.g. **Mamba / Selective State Space Models**, **RWKV**, and **RetNet**) do **not** use raw cumulative sums. They introduce an **input-dependent decay gate** ($\exp(-\Delta_t A)$) that allows the model to selectively decay past tokens and overwrite memory.
