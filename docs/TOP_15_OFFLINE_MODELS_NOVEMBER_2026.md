# Top 15 Offline Frontier Model Candidates for WEFT Streaming Deconstruction

Curated from `/home/akarta/Downloads/offline-models-november-2026-hf-links-9251b13a.txt` (checked 21 November 2026 snapshot).
Selected as the post-pipeline targets following the immediate four-model deconstruction run:
1. **Model A**: `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` (DeepSeek-V2 MoE + MLA)
2. **Model B**: `mistralai/Mixtral-8x7B-Instruct-v0.1` (8-Expert Sparse MoE)
3. **Model C**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` (Frontier Mathematical Reasoning)
4. **Model D**: `openai/whisper-large-v3` (Multimodal Audio Seq2Seq)

---

## 1. Top 15 Curated Fleet Candidates

The 15 models below represent premier architectural, algorithmic, and multimodal diversity across open-weights frontier systems:

| # | Model Identifier | Architecture Class | Hugging Face Repository | Parameters (Total / Active) | Strategic Deconstruction Target |
|---|---|---|---|---|---|
| **01** | **Qwen3-Coder-Next** | Sparse MoE Coding Specialist | [`Qwen/Qwen3-Coder-Next`](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 80B Total / 3B Active | Sparse MoE token routing bitboards with $262\text{K}$ native context window. |
| **02** | **Qwen3.8-Flash-Next** | Recurrent Linear Attention + MTP | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 180B Total / 6B Active | Linear-attention recurrence kernels with multi-token prediction (MTP) heads. |
| **03** | **DeepSeek-V4.1-Flash-1017** | Sparse MLA + MoE Frontier | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | 100B+ MoE | High-speed frontier MoE benchmarked at 33 on Artificial Analysis index. |
| **04** | **Nemotron-3.5-Lightning-30B-A3B** | Hybrid Mamba-2 SSM + Sparse MoE | [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) | 30B Total / 3B Active | State-space SSM recurrent state factorization with up to $1\text{M}$ context. |
| **05** | **gpt-oss-120b** | Frontier Sparse MoE Transformer | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) | 117B Total / 5.1B Active | OpenAI open-weights sparse MoE architecture with Harmony conversation format. |
| **06** | **Inkling-2-Mini** | Looped-Depth Transformer | [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) | 26.4B Looped | Dynamic middle-layer recurrence; ideal for frozen sub-bit cyclic bitboards. |
| **07** | **Olmo-3.1-32B-Think** | Transparent Open-Science Reasoning | [`allenai/Olmo-3.1-32B-Think`](https://huggingface.co/allenai/Olmo-3.1-32B-Think) | 32B Dense | Fully transparent checkpoints and training trajectories for spectral decay audits. |
| **08** | **Nanbeige4.3-33B** | Dense Workstation Reasoning | [`Nanbeige/Nanbeige4.3-33B`](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) | 33B Dense | High-efficiency 33B dense architecture scoring 20 on Artificial Analysis index. |
| **09** | **Ministral-3-14B-Instruct-2512** | Dense Edge/Workstation Frontier | [`mistralai/Ministral-3-14B-Instruct-2512`](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512) | 14B Dense | Sub-20B edge flagship with interleaved sliding window and global attention. |
| **10** | **Granite-4.2-30B** | Enterprise MoE Workhorse | [`ibm-granite/granite-4.2-30b`](https://huggingface.co/ibm-granite/granite-4.2-30b) | 30B MoE | Code/reasoning fine-tuned enterprise MoE; testbed for high-sparsity routing. |
| **11** | **Nemotron-3.5-Bolt-E4B** | Gemma-4 Derived Recurrent Agent | [`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16) | 8B Total / 4B Active | Ultra-efficient edge agent foundation designed for $\le 4\text{ GB}$ hardware footprints. |
| **12** | **Muse-Glimmer-30B** | Low-Overhead Generative Reasoner | [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B) | 30B Dense | Native ExecuTorch PTE / GGUF compatible architecture for edge deployment. |
| **13** | **GLM-5.3-Flash** | Bilingual Frontier Sparse MoE | [`THUDM/glm-5.3-flash`](https://huggingface.co/THUDM/glm-5.3-flash) | 60B+ Sparse MoE | Fast Chinese-English reasoning architecture scoring 28 on Artificial Analysis index. |
| **14** | **Qwen3.8-27B** | High-Compute Dense Foundation | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B Dense | Balanced workstation dense target for comprehensive language & code benchmarks. |
| **15** | **FLUX-3-Image-Klein** | Diffusion-Transformer Vision Backbone | [`black-forest-labs/FLUX.3-Image-Klein`](https://huggingface.co/black-forest-labs/FLUX.3-Image-Klein) | 5.6B DiT | Multimodal diffusion-transformer latent projection and sub-bit DiT block verification. |

---

## 2. Ingestion & Execution Pipeline Protocols

All subsequent deconstruction targets adhere strictly to the established WEFT operational guardrails:
1. **$O(1)$ Host Memory Bounds**: Randomized low-rank decomposition (`torch.svd_lowrank`) per layer/expert block with memory scratch $<64\text{ MB}$, eliminating LAPACK memory blowup.
2. **Strict Disk Safety Floor**: Shard streaming with immediate uncompressed `.safetensors` purging upon `.weft` serialization, strictly guaranteeing $\ge 700\text{ GB}$ free space on `/run/media/akarta/AI`.
3. **Formal Invariant Certification**: Automated verification against Lean 4 proof modules (`formal/MoERoutingInvariants.lean`, `formal/TernaryQuantizationBounds.lean`, `formal/SubBitAttentionBounds.lean`).
4. **Three-Way Artifact Parity**: Consolidated GGUF generation, live autoregressive decode latency verification, and matching `.md` / `.tex` / `.pdf` research publications.
