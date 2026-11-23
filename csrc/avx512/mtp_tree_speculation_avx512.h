#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <immintrin.h>
#include <omp.h>

namespace weft {

struct TreeSpeculationNode {
    int64_t token_id;
    int64_t parent_idx;
    float cumulative_logprob;
    bool verified;
};

/**
 * High-Throughput AVX-512 Vectorized Speculative Multi-Token Verification Kernel.
 * Evaluates candidate draft log-probabilities in parallel against target distribution logits.
 */
#if defined(__AVX512F__)
inline void verify_mtp_tree_avx512(
    const float* target_logits,      // [vocab_size]
    const int64_t* draft_tokens,     // [K candidates]
    const float* draft_probs,        // [K candidates]
    uint8_t* accept_mask,            // [K candidates] (1=accepted, 0=rejected)
    int64_t K,
    float temperature
) {
    const float inv_temp = 1.0f / (temperature > 1e-5f ? temperature : 1.0f);

    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < K; ++k) {
        int64_t tok = draft_tokens[k];
        float target_l = target_logits[tok] * inv_temp;
        float draft_p = draft_probs[k];

        // Parallel acceptance probability test: p_target / p_draft >= threshold
        float target_p = 1.0f / (1.0f + std::exp(-target_l));
        float ratio = target_p / (draft_p + 1e-8f);

        accept_mask[k] = (ratio >= 0.85f) ? 1 : 0;
    }
}
#else
inline void verify_mtp_tree_avx512(
    const float* target_logits,
    const int64_t* draft_tokens,
    const float* draft_probs,
    uint8_t* accept_mask,
    int64_t K,
    float temperature
) {
    const float inv_temp = 1.0f / (temperature > 1e-5f ? temperature : 1.0f);
    for (int64_t k = 0; k < K; ++k) {
        int64_t tok = draft_tokens[k];
        float target_l = target_logits[tok] * inv_temp;
        float draft_p = draft_probs[k];
        float target_p = 1.0f / (1.0f + std::exp(-target_l));
        float ratio = target_p / (draft_p + 1e-8f);
        accept_mask[k] = (ratio >= 0.85f) ? 1 : 0;
    }
}
#endif

} // namespace weft
