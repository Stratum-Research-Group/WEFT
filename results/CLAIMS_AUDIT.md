# Empirical Claims Audit & Hardware Artifact Mapping

This audit maps every performance metric and system characteristic presented in Papers 354–365 to its verified physical hardware benchmark artifact on the AMD Ryzen 5 7600X and NVIDIA GeForce RTX 5060 Ti Blackwell (SM 12.0 / 16GB GDDR7).

---

## 1. Clustered MoE Dispatch Routing (Papers 360, 363)
- **Claimed in Literature:** 2-tier hierarchical clustering (16 clusters, 8 experts/cluster) accelerates routing across 128 experts by block-sparse pruning.
- **Physical Hardware Artifact:** `results/fused_moe_router_rtx5060ti.json` (evaluated via CUDA events across 200 iterations).
- **Physical Measurements:**
  - $B=1$: PyTorch flat GEMM is fastest at $31.13\ \mu\text{s}$.
  - Triton Flat Router: $184.57\ \mu\text{s}$ ($B=1$) $\to 1,642.77\ \mu\text{s}$ ($B=256$).
  - Triton Fused Clustered Router: $44.58\ \mu\text{s}$ ($B=1$) $\to 165.85\ \mu\text{s}$ ($B=256$).
  - **Measured Speedup:** Clustered Triton vs Flat Triton is **$4.14\times$ at $B=1$**, **$8.29\times$ at $B=64$**, and **$9.90\times$ at $B=256$**.
- **Correction:** Clarified in paper text that for single-token micro-batches ($B=1$), cuBLAS flat matrix multiplication is faster due to negligible dimensional overhead, while Triton clustered dispatch provides up to $9.9\times$ speedup at higher batch sizes.

---

## 2. Speculative Tree Verification & End-to-End Decoding (Papers 356, 358)
- **Claimed in Literature:** Multi-candidate tree speculation achieves $4.00\times$ step speedup against teacher states.
- **Physical Hardware Artifact:** `results/speculative_real_qwen_coder.json` (evaluated on `Qwen2.5-Coder-0.5B` draft and `Qwen2.5-Coder-1.5B` target).
- **Physical Measurements:**
  - Baseline Target Greedy: $46.8\text{ tok/s}$.
  - Speculative $k=2$: $34.7\text{ tok/s}$ ($0.74\times$ wall-clock speedup, **$93.3\%$ acceptance rate**).
  - Speculative $k=4$: $39.3\text{ tok/s}$ ($0.84\times$ wall-clock speedup, **$88.8\%$ acceptance rate**).
- **Correction:** The step acceptance rate is empirically confirmed at high fidelity ($88.8\% - 93.3\%$), but on a single PCIe GPU without CUDA graph capture, memory-bandwidth and sequential draft generation overhead result in $0.84\times$ wall-clock execution relative to pure greedy decoding.

---

## 3. Streaming NVMe Sharded Deconstruction (Papers 354, 357, 361)
- **Physical Hardware Artifacts:** 
  - `results/frontier_1t_streaming_benchmark.json` (16-shard run: $0.83\text{ MB/s}$ throughput, $778.0\text{ MB}$ peak RSS).
  - `results/frontier_1t_streaming_64shards.json` (64-shard run: $0.97\text{ MB/s}$ throughput, $777.9\text{ MB}$ peak RSS, $49.5\text{ MB}$ drift).
- **Drive Safety Verification:** Zero disk creep confirmed; `/run/media/akarta/AI` sustained at **$\ge 770.4\text{ GB free}$** ($\ge 700\text{ GB}$ safety floor).
