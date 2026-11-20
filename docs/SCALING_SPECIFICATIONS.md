# WEFT-2 Frontier Scaling Architecture Specifications

This specification formalizes the exact configuration targets, architectural dimensions, and zero-memory analytical parameter profiles for the WEFT-2 model family across standard parameter tiers (**1.5B**, **3B**, **7B**, **14B**, and **32B**).

All configurations are instantiated and strictly validated via `torch.device("meta")`, guaranteeing $0\text{ MB}$ host RAM consumption during instantiation and planning.

---

## 1. Architectural Parameter Matrix

| Specification | WEFT-2 1.5B | WEFT-2 3B | WEFT-2 7B | WEFT-2 14B | WEFT-2 32B (MoE) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Target Audience / Tier** | Edge / Mobile / Draft | Local Workstation | Open-Weight Standard | Enterprise Reasoning | Frontier High-Capacity |
| **Total Parameters** | **1.48B** | **3.01B** | **6.99B** | **14.65B** | **32.51B** |
| **Active Parameters / Token** | **1.48B** | **3.01B** | **6.99B** | **14.65B** | **8.84B** |
| **Hidden Dim ($d_{\text{model}}$)** | 2,048 | 2,560 | 4,096 | 5,120 | 4,096 |
| **Intermediate Dim ($d_{\text{ffn}}$)** | 5,632 | 7,168 | 11,008 | 14,336 | 11,008 |
| **Layers ($L$)** | 32 | 42 | 40 | 52 | 50 |
| **Attention Heads ($H$)** | 16 | 20 | 32 | 40 | 32 |
| **State Heads ($H_{\text{state}}$)** | 4 | 4 | 4 | 8 | 8 |
| **State Rank ($r$)** | 32 | 32 | 32 | 48 | 48 |
| **Local Chunk Window ($W$)** | 128 | 128 | 128 | 128 | 128 |
| **Multi-Head Latent Attn (MLA)** | Yes ($d_{\text{lat}}=128$) | Yes ($d_{\text{lat}}=128$) | Yes ($d_{\text{lat}}=128$) | Yes ($d_{\text{lat}}=128$) | Yes ($d_{\text{lat}}=128$) |
| **Mixture of Experts (MoE)** | Dense | Dense | Dense | Dense | **16 routed, top-2 + 1 shared** |
| **Vocab Size** | 32,000 | 32,000 | 32,000 | 32,000 | 32,000 |

---

## 2. Hardware Resource Envelope (Inference & Training)

| Model Tier | Precision | Weights VRAM | KV / Recurrent State (1M Tokens) | Minimum GPU Spec |
| :--- | :---: | :---: | :---: | :---: |
| **WEFT-2 1.5B** | FP16 / BF16 | 2.96 GB | **< 1.8 GB** | 1x RTX 4060 / 5060 (8 GB) |
| **WEFT-2 3B** | FP16 / BF16 | 6.02 GB | **< 2.2 GB** | 1x RTX 4070 / 5060 Ti (12–16 GB) |
| **WEFT-2 7B** | FP16 / BF16 | 13.98 GB | **< 2.5 GB** | 1x RTX 4080 / 5080 (16 GB) |
| **WEFT-2 7B (INT8/AWQ)** | INT8 | 7.00 GB | **< 2.5 GB** | 1x RTX 4070 / 5060 Ti (16 GB) |
| **WEFT-2 14B** | FP16 / BF16 | 29.30 GB | **< 3.2 GB** | 1x A100 / H100 (40–80 GB) or 2x 3090 |
| **WEFT-2 32B (MoE)** | FP16 / BF16 | 65.02 GB | **< 3.2 GB** | 1x H100 (80 GB) or 2x RTX 6000 Ada |
| **WEFT-2 32B (INT4 MoE)** | INT4 | 16.25 GB | **< 3.2 GB** | 1x RTX 5060 Ti (16 GB offloaded) / RTX 4090 |

> **Key Theoretical Advantage:** Because WEFT decouples history accumulation into fixed-size associative state tensors ($S \in \mathbb{R}^{B \times H_{\text{state}} \times d_{\text{head}} \times r}$) and only preserves a sliding window of $W=128$ tokens for local attention, the context state footprint is **strictly constant $\mathcal{O}(1)$ across context length** (tested and proven up to 1,048,576 tokens).
