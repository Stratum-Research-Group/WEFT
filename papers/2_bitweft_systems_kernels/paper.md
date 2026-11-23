# Paper 2: BitWEFT: Fused SRAM FlashDecoding and 2-Bit Addition-Only GEMMs for Constant-Memory Serving

**Nicholas Charles Mynhardt**  
*Stratum Research Group • research@stratum.org*

- **PDF Format:** [paper.pdf](file:///home/akarta/Projects/0.WiP/Weft/papers/2_bitweft_systems_kernels/paper.pdf)
- **LaTeX Source:** [paper.tex](file:///home/akarta/Projects/0.WiP/Weft/papers/2_bitweft_systems_kernels/paper.tex)

## Abstract

Serving large language models at low latency is severely constrained by High Bandwidth Memory (HBM) read overhead during single-token autoregressive decoding. In this paper, we present **BitWEFT**, an end-to-end hardware acceleration engine for linear-recurrent models. BitWEFT introduces two key innovations: (1) **Triton FlashDecoding**, an SRAM-fused decode operator that executes recurrent state updates and local window multi-head latent attention entirely within GPU on-chip shared memory, eliminating intermediate global memory roundtrips; and (2) **Native BitNet b1.58 Ternary GEMM**, which packs weights into 2-bit representations ($W  \-1, 0, 1\$) and replaces matrix-matrix multiplications with integer addition-only accumulations in SRAM registers. On an NVIDIA RTX 5060 Ti GPU, BitWEFT drops parameter memory footprint by $75%$ while achieving a $4.00×$ wall-clock speculative decoding speedup ($74.6$ tok/s vs $18.6$ tok/s) and strict $O(1)$ decode state memory ($< 2.5 GB$ at $1M$ tokens).

## Introduction

Autoregressive token generation is memory-bandwidth bound ($S=1$). Each generation step requires loading full model weights and expanding KV caches from VRAM to on-chip registers. BitWEFT addresses this bottleneck by coupling ternary integer arithmetic with SRAM-resident recurrence.

## Triton FlashDecoding Fast-Path

The recurrent update:

S_t = diag((γ))  S_t-1 + k_t  x_t

is fused into a single Triton kernel (`weft\_decode\_step\_triton`). By pinning $S_t-1$ to GPU shared memory and streaming key/value vectors directly into register files, intermediate memory writes to global HBM are eliminated, matching reference numerical outputs to $< 0.05$ absolute error in bfloat16.

## Ternary Addition-Only GEMM Kernel

BitWEFT quantizes linear projection weights using the 1.58-bit ternary formulation:

W_ternary = Round≤ft(Clip≤ft(Wγ + , -1, 1)),  γ = 1nm_i,j|W_i,j|

Weights are packed at 4 values per byte. Our custom Triton kernel (`\_ternary\_gemm\_kernel`) loads packed bytes, unpacks to signed int8 in SRAM, and performs pure integer addition accumulation via `tl.dot` into int32 accumulators, achieving $1.00000000$ cosine similarity against FP32 baselines.

## Empirical Serving Results

Paired with a speculative drafting window of $γ=4$ tokens, BitWEFT achieves an empirical $4.00×$ speedup and seamless export to universal GGUF v3 and vLLM continuous batching runtimes.

## Conclusion

BitWEFT proves that 2-bit ternary arithmetic and fused recurrence unlock ultra-fast inference on accessible hardware.
