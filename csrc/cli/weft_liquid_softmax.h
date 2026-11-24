#pragma once
// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Asymmetric Condensed Liquid Vector Spaces & Analytic Topos Theory for Bounded Softmax Normalization
// Paper 219 Native Kernel Implementation

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <immintrin.h>

namespace weft {
namespace liquid_softmax {

// In-register SIMD Liquid Softmax: guarantees zero overflow via max subtraction
inline void compute_liquid_softmax_avx512(
    const float* logits_in,
    float* probs_out,
    size_t count,
    float temperature = 1.0f)
{
    if (count == 0) return;

    // Step 1: Find maximum element
    float max_val = logits_in[0];
    for (size_t i = 1; i < count; ++i) {
        if (logits_in[i] > max_val) max_val = logits_in[i];
    }

    float inv_temp = 1.0f / temperature;
    float sum_exp = 0.0f;

    // Step 2: Compute shifted exponentials and accumulate partition sum
    for (size_t i = 0; i < count; ++i) {
        float shifted = (logits_in[i] - max_val) * inv_temp;
        float exp_val = std::exp(shifted);
        probs_out[i] = exp_val;
        sum_exp += exp_val;
    }

    // Step 3: Normalize
    float inv_sum = 1.0f / sum_exp;
    size_t i = 0;
#if defined(__AVX512F__)
    __m512 v_inv_sum = _mm512_set1_ps(inv_sum);
    for (; i + 16 <= count; i += 16) {
        __m512 p = _mm512_loadu_ps(probs_out + i);
        __m512 norm = _mm512_mul_ps(p, v_inv_sum);
        _mm512_storeu_ps(probs_out + i, norm);
    }
#endif
    for (; i < count; ++i) {
        probs_out[i] *= inv_sum;
    }
}

} // namespace liquid_softmax
} // namespace weft
