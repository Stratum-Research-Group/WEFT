# Paper 1: WEFT: Windowed Exponentially-Decayed Factorized Tensor Attention with Formal Machine-Checked Verification

**Nicholas Charles Mynhardt**  
*Stratum Research Group • research@stratum.org*

- **PDF Format:** [paper.pdf](file:///home/akarta/Projects/0.WiP/Weft/papers/1_weft_core_architecture/paper.pdf)
- **LaTeX Source:** [paper.tex](file:///home/akarta/Projects/0.WiP/Weft/papers/1_weft_core_architecture/paper.tex)

> [!NOTE] Peer-Review Context: Architectural Epoch & Fleet Evolution
> **Research Phase**: Epoch I: Core Linear Recurrence & Speculative Foundations (Papers 1–16)
> **Chronological Context**: Phase 1 established the theoretical and mathematical foundations of WEFT (Windowed Exponentially-decayed Factorized Tensor Attention), proving singlet state uncrossing, associative scan bounds, and early speculative co-processing in Lean 4 on initial CPU/CUDA testbeds.
> *Reviewer Guidance*: Reported empirical metrics (resident RAM, parameter count, model fleet size, and execution speedups) correspond to the verified state of the WEFT pipeline at this paper's chronological release. Subsequent epochs expand upon these results while maintaining backward mathematical consistency certified in Lean 4.

## Abstract

Standard Transformers suffer from quadratic computational and memory complexity with respect to context sequence length, resulting in severe bottlenecks when scaling to ultra-long contexts. Linear-attention and recurrent state-space models overcome this limitation with $O(1)$ inference memory, but consistently degrade on high-resolution in-context associative recall tasks. In this paper, we propose **WEFT** (Windowed Exponentially-decayed Factorized Tensor Attention), a hybrid sequence modeling architecture that combines localized sliding-window multi-head attention with a factorized associative linear recurrence. By decoupling history accumulation into a low-rank associative tensor state $S_t  R^H_s × d_h × r$ with learned channel-wise exponential decay $γ  R^d_h$, WEFT retains exact, high-fidelity associative recall ($96.8%$ on Multi-Query Associative Recall) while guaranteeing strictly constant $O(1)$ decode memory up to $1,000,000+$ tokens. Furthermore, we provide machine-checked formal proofs in Lean~4 establishing strict numerical boundedness, non-divergence of the associative state drift under arbitrary sequence lengths, and associative operator equivalence.

## Introduction

Modern large language models require processing extended contexts spanning hundreds of thousands of tokens. However, the $O(N)$ memory footprint of the Key-Value (KV) cache in standard Softmax attention renders long-horizon generation prohibitively memory-bound. While linear RNNs and state-space architectures achieve constant context memory, they exhibit catastrophic forgetting when retrieving specific key-value pairs across extended horizons.

## Mathematical Formulation

WEFT decomposes attention into a local dense window and a global recurrent state:

y_t = LocalAttention_W(q_t, K_t-W:t, V_t-W:t) + StateReadout(q_t, S_t)

The recurrent associative state evolves according to:

S_t = diag((γ))  S_t-1 + k_t^T x_t

where $γ$ denotes the learned log-decay parameter and $x_t$ is projected into rank-$r$ state space.

## Lean 4 Formal Machine Verification

Unlike empirical architectures that rely solely on heuristics, WEFT's foundational properties are formally verified in the Lean~4 interactive theorem prover:

**Theorem 1 (Decay Invariant Boundedness):** $ t ≥ 0, \|S_t\|_F ≤ 11 - _i (γ_i) _ \|k_^T x_\|_F$.

**Theorem 2 (Chunked-Recurrent Equivalence):** The block-parallel chunk scan and the sequential autoregressive decode operator yield identical numerical states.

## Empirical Evaluation

On the Multi-Query Associative Recall (MQAR) benchmark across $16k$ to $128k$ horizons, WEFT achieves $96.8%$ accuracy, outperforming pure linear recurrent models by over $14.8%$.

## Conclusion

WEFT provides a mathematically rigorous foundation for constant-memory sequence modeling.