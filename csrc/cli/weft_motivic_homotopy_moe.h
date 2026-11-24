#pragma once
// weft_motivic_homotopy_moe.h
// Milestone Paper 234: Native AVX-512 Motivic Homotopy MoE Routing & Retraction
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>
#include <vector>
#include <algorithm>

namespace weft {

struct MotivicMoEConfig {
    size_t num_experts;
    size_t top_k;
    float damping_lambda;
};

// Apply motivic Hopf retraction projection to input logits:
// Pi(x)_i = x_i / (1 + lambda * sum(x^2))
inline void avx512_motivic_hopf_project(
    const float* __restrict__ in_logits,
    float* __restrict__ out_projected,
    size_t num_experts,
    float damping_lambda
) {
    // 1. Compute L2 norm sum(x^2) using AVX-512
    __m512 v_sq_sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= num_experts; i += 16) {
        __m512 v = _mm512_loadu_ps(in_logits + i);
        v_sq_sum = _mm512_fmadd_ps(v, v, v_sq_sum);
    }
    float sq_sum = _mm512_reduce_add_ps(v_sq_sum);
    for (; i < num_experts; ++i) {
        sq_sum += in_logits[i] * in_logits[i];
    }

    float denom = 1.0f + damping_lambda * sq_sum;
    float scale = (denom > 1e-7f) ? (1.0f / denom) : 1.0f;
    __m512 v_scale = _mm512_set1_ps(scale);

    // 2. Vectorized scaling
    i = 0;
    for (; i + 16 <= num_experts; i += 16) {
        __m512 v = _mm512_loadu_ps(in_logits + i);
        __m512 scaled = _mm512_mul_ps(v, v_scale);
        _mm512_storeu_ps(out_projected + i, scaled);
    }
    for (; i < num_experts; ++i) {
        out_projected[i] = in_logits[i] * scale;
    }
}

// Select top-k expert indices based on projected scores
inline void select_topk_experts(
    const float* __restrict__ scores,
    size_t num_experts,
    size_t top_k,
    size_t* __restrict__ out_indices
) {
    std::vector<std::pair<float, size_t>> scored_indices(num_experts);
    for (size_t i = 0; i < num_experts; ++i) {
        scored_indices[i] = {scores[i], i};
    }

    size_t k = (top_k <= num_experts) ? top_k : num_experts;
    std::partial_sort(
        scored_indices.begin(),
        scored_indices.begin() + k,
        scored_indices.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    for (size_t i = 0; i < k; ++i) {
        out_indices[i] = scored_indices[i].second;
    }
}

} // namespace weft
