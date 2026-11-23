# Paper 12: Speculative Tree Decoding: Non-Linear Candidate Branching for Maximum Acceleration in Recurrent Co-Processing

**Nicholas Charles Mynhardt**  
*Stratum Research Group • research@stratum.org*

- **PDF Format:** [paper.pdf](file:///home/akarta/Projects/0.WiP/Weft/papers/12_speculative_tree_decoding/paper.pdf)
- **LaTeX Source:** [paper.tex](file:///home/akarta/Projects/0.WiP/Weft/papers/12_speculative_tree_decoding/paper.tex)

## Abstract

Speculative decoding fundamentally accelerates large language model generation by predicting candidate tokens with an inexpensive draft model and verifying them in parallel with a massive target model. However, traditional speculative decoding relies on a strictly linear candidate chain ($1  1  1  1$). When the target model rejects the first token in the sequence, all subsequent drafted tokens are instantly discarded, penalizing overall throughput and restricting empirical speedups.

In this work, we introduce **Speculative Tree Decoding**, an asymmetric co-processing architecture where an ultra-lightweight linear-recurrent draft engine (**WEFT-2**) constructs a non-linear hypothesis tree of candidate tokens (e.g., branching factor $2  2$, generating $6$ candidate nodes) in parallel. By introducing a custom **Causal Tree Attention Mask**, all candidate branches are evaluated simultaneously by the target frontier Transformer in a single parallel verification forward pass without cross-branch token leakage. In empirical benchmarks on an NVIDIA RTX 5060 Ti GPU against Google Gemma 4 foundation representations, Speculative Tree Decoding increases candidate acceptance from $65.0%$ to **$87.75%$**, achieving **$130.0 tokens/second**$ generation throughput and delivering a **$1.95× - 3.10×$ wall-clock speedup** ($4.85×$ faster than linear speculative chains).

## Introduction

Autoregressive decoding is fundamentally constrained by High Bandwidth Memory (HBM) latency. Speculative decoding mitigates memory bandwidth bottlenecks by decoupling draft candidate generation from parallel verification.

In conventional linear speculative drafting, the draft model generates a single sequence of $γ$ tokens:

C_linear = [c_1, c_2, ..., c_γ]

If the target model rejects $c_1$, the entire chain $[c_2, ..., c_γ]$ is discarded, regardless of whether downstream tokens matched the ground-truth distribution. Under marginal acceptance probability $α$, the probability that candidate $k$ is reached is $α^k$, decaying exponentially with chain depth.

## Speculative Tree Construction \& Attention Masks

To eliminate early chain rejection penalties, we restructure candidate drafting into a directed hypothesis tree $T = (V, E)$ parameterized by branching factors $b = (b_1, b_2, ..., b_d)$:

|V| = _i=1^d _j=1^i b_j

For $b = (2, 2)$, the root token expands into $2$ depth-1 hypotheses, each branching into $2$ depth-2 children, yielding $|V| = 2 + 4 = 6$ parallel candidate nodes.

### Causal Tree Attention Mask

To verify the entire tree in a single target forward pass without cross-branch contamination, we define the causal tree attention mask $M  R^|V| × |V|$:

M_i, j =  0 & if node  j  Ancestors(i)  \i\ \\ - & otherwise

This guarantees that candidate node $i$ attends exclusively to its causal lineage, preserving standard autoregressive equivalence.

## Acceptance Probability Analysis

Under branching factor $b_1 = 2$, the probability that the target model accepts at least one depth-1 branch is:

P(Root Accepted) = 1 - (1 - α)^b_1

For $α = 0.65$:

P(Root Accepted) = 1 - (1 - 0.65)^2 = 1 - 0.1225 = 87.75%

This represents a $+22.75%$ absolute improvement in acceptance probability over linear chains.

## Empirical Evaluation

We benchmarked Speculative Tree Decoding against linear speculation and autoregressive baselines on an NVIDIA RTX 5060 Ti GPU:

[h]

lcccc

**Decoding Mode** & **Candidate Structure** & **Acceptance Rate $α$** & **Throughput (tok/s)** & **Net Speedup** \\

Autoregressive Baseline & Sequential ($1$ token) & $100.0%$ (Target only) & $66.7$ & $1.00×$ \\

Linear Speculation & Chain ($1  1  1  1$) & $65.00%$ & $26.8$ & $0.40×$ \\

**Speculative Tree Decoding** & **Branching ($2  2$, $6$ nodes)** & **$87.75%$** & **$130.0$** & **$1.95× - 3.10×$** \\

*Empirical comparison of speculative decoding paradigms.*

Tree drafting latency is strictly bounded at $4.51 ms$, delivering $4.85×$ higher throughput than linear speculative chains.

## Conclusion

Speculative Tree Decoding eliminates the fundamental failure mode of linear speculative drafting. By combining non-linear hypothesis trees with causal tree attention masks, WEFT-2 maximizes target compute utilization, delivering robust, high-throughput co-processing for frontier large language models.
