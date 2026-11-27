# WEFT Foundation Model Fleet Physical Shard Inventory

**Stratum Research Group • Nicholas Charles Mynhardt • Apache-2.0**

### Storage Subsystem: `/run/media/akarta/AI/models`

**Total Converted Model Families:** 14
**Aggregate Resident Sub-Bit Storage:** 4.28 GB

| Foundation Model Architecture | Weft Shards | Consolidated GGUF | Resident Size (MB) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **deepseek-coder-v2-lite-subbit** | 4 shards | Yes (model_consolidated_subbit.gguf) | 668.4 MB | **ONLINE / ZERO-COPY MMAP** |
| **deepseek-r1-distill-qwen-32b-subbit** | 8 shards | Yes (model_consolidated_subbit.gguf) | 310.4 MB | **ONLINE / ZERO-COPY MMAP** |
| **goedel-prover-32b-subbit** | 14 shards | Yes (model_consolidated_subbit.gguf) | 311.7 MB | **ONLINE / ZERO-COPY MMAP** |
| **lfm3-vl-2b-subbit** | 1 shards | Standalone Weft | 24.4 MB | **ONLINE / ZERO-COPY MMAP** |
| **llama-3.3-70b-subbit** | 30 shards | Yes (llama-3.3-70b-shards1-12.gguf, llama-3.3-70b-shards1-6.gguf, model_consolidated_subbit.gguf) | 597.2 MB | **ONLINE / ZERO-COPY MMAP** |
| **mixtral-8x7b-subbit** | 19 shards | Yes (model_consolidated_subbit.gguf) | 545.5 MB | **ONLINE / ZERO-COPY MMAP** |
| **musicgen-small-subbit** | 1 shards | Yes (model_consolidated_subbit.gguf) | 32.4 MB | **ONLINE / ZERO-COPY MMAP** |
| **qwen-2.5-72b-subbit** | 37 shards | Yes (model_consolidated_subbit.gguf) | 482.9 MB | **ONLINE / ZERO-COPY MMAP** |
| **qwen2-vl-7b-subbit** | 5 shards | Yes (model_consolidated_subbit.gguf) | 124.7 MB | **ONLINE / ZERO-COPY MMAP** |
| **qwen2.5-coder-14b-subbit** | 6 shards | Yes (model_consolidated_subbit.gguf) | 164.0 MB | **ONLINE / ZERO-COPY MMAP** |
| **qwen2.5-coder-32b-subbit** | 14 shards | Yes (model_consolidated_subbit.gguf) | 309.9 MB | **ONLINE / ZERO-COPY MMAP** |
| **qwen2.5-coder-7b-subbit** | 4 shards | Yes (model_consolidated_subbit.gguf) | 100.8 MB | **ONLINE / ZERO-COPY MMAP** |
| **qwen2.5-math-72b-subbit** | 37 shards | Yes (model_consolidated_subbit.gguf) | 481.4 MB | **ONLINE / ZERO-COPY MMAP** |
| **whisper-large-v3-subbit** | 3 shards | Yes (model_consolidated_subbit.gguf) | 127.0 MB | **ONLINE / ZERO-COPY MMAP** |

### Sharding & Execution Architecture
1. **Zero-Copy POSIX Memory-Mapping:** All `.weft` shards are formatted with 64-byte AVX-512 cacheline alignment and loaded via `mmap` with zero heap allocation overhead.
2. **Hardware Concurrency:** Linear weight paths are evaluated via in-register AVX-512 VPOPCNTDQ on the AMD Ryzen 5 7600X CPU, with latent speculative drafting offloaded to the NVIDIA GeForce RTX 5060 Ti GPU (< 48 MB VRAM).
