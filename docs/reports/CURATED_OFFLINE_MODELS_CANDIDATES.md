# Curated Offline Models for WEFT Sub-Bit Quantization Testing

Curated from `/home/akarta/Downloads/offline-models-november-2026-hf-links-9251b13a.txt` (November 2026 snapshot).
Selected across 5 distinct architectural frontiers: **Extreme Scale Reasoning, Formal Lean Verification, Coding Agents, Recurrent/Linear Attention, and Edge Vision/Audio**.

---

## 1. Top 10 Curated Model Candidates

| # | Model Identifier | Hugging Face Repository | Architecture / Parameters | Research & Quantization Motivation |
|---|------------------|-------------------------|---------------------------|-----------------------------------|
| **01** | **Llama-3.3-70B-Instruct** | [`meta-llama/Llama-3.3-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | 70B Dense Transformer (141 GB FP16) | Gated access verified. Prime target for streaming sub-bit quantization down to $\approx 2.2$ GB ($0.25$ BPW). |
| **02** | **Goedel-Prover-V2-32B** | [`Goedel-LM/Goedel-Prover-V2-32B`](https://huggingface.co/Goedel-LM/Goedel-Prover-V2-32B) | 32B Dense Formal Reasoning | Specialized for Lean 4 theorem proving. Ideal companion for our internal Lean 4 invariant verifier. |
| **03** | **Qwen3-Coder-Next** | [`Qwen/Qwen3-Coder-Next`](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 80B Total / 3B Active MoE (262K ctx) | Extreme sparse MoE coding specialist. Testbed for routing-bounded sub-bit bitboard execution. |
| **04** | **Qwen3.8-Flash-Next** | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 180B Total / 6B Active Linear+MTP | Recurrent linear-attention + Multi-Token Prediction (MTP). Validates WEFT's chunk scan and sub-bit tree kernels. |
| **05** | **Nemotron-3.5-Lightning-30B-A3B** | [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) | 30B Total / 3B Active Hybrid Mamba/MoE | Hybrid state-space + attention architecture with up to 1M context. Perfect for Sub-Bit PagedAttention. |
| **06** | **Inkling-2-Mini** | [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) | 26.4B Looped-Depth Transformer | November 2026 release testing repeated middle layers (compute looping) with sub-bit frozen bitboards. |
| **07** | **gpt-oss-120b** | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) | 117B Total / 5.1B Active Sparse MoE | Open weights release from OpenAI with Harmony conversation format. Stress-tests streaming sharded conversion. |
| **08** | **Olmo-3.1-32B-Think** | [`allenai/Olmo-3.1-32B-Think`](https://huggingface.co/allenai/Olmo-3.1-32B-Think) | 32B Open Scientific Reasoning | Fully open weights, data, and training trajectories. Ideal for white-box spectral decay analysis. |
| **09** | **LFM3-VL-2B** | [`LiquidAI/LFM3-VL-2B`](https://huggingface.co/LiquidAI/LFM3-VL-2B) | 1.92B Liquid Multimodal (Vision+Lang) | Validates our new `SubBitMultimodalProjector` with continuous visual frame token embeddings. |
| **10** | **Nemotron-3.5-Bolt-E4B** | [`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16) | 8B Total / 4B Effective (Gemma-4 derived) | Edge-agent specialist. Directly comparable against our existing Gemma-4 E2B/E4B sub-bit quantization baseline. |

---

## 2. Storage & Execution Strategy

Using our **Streaming Sharded Quantization Pipeline** (`weft/stream_quantizer.py`) with randomized low-rank SVD (`torch.svd_lowrank`):
- Each model shard is downloaded into `/run/media/akarta/AI/models/<model>-subbit/.stream_staging/`.
- Decomposed into bipolar bitboards and scale vectors without dense LAPACK scratch blowup ($<39$ MB memory workspace).
- The raw shard is purged immediately upon `.weft` container write.
- Peak disk usage remains bounded to a single uncompressed shard ($\le 5$ GB), preserving the **703 GB available** capacity on the AI SSD.
