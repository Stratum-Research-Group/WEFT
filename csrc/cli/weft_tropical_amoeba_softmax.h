#pragma once
// weft_tropical_amoeba_softmax.h
// Milestone Paper 247: Native AVX-512 Tropical Amoeba Softmax Normalization
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>
#include <algorithm>

namespace weft {

// Vectorized piecewise-linear tropical amoeba softmax:
// 1. m = max_j s_j
// 2. phi_i = max(0.0f, 1.0f + alpha * (s_i - m))
// 3. p_i = phi_i / sum(phi)
inline void avx512_tropical_amoeba_softmax(
    const float* __restrict__ in_logits,
    float* __restrict__ out_probs,
    size_t count,
    float alpha
) {
    if (count == 0) return;

    // 1. Compute max logit using AVX-512
    __m512 v_max = _mm512_set1_ps(-1e30f);
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 v = _mm512_loadu_ps(in_logits + i);
        v_max = _mm512_max_ps(v_max, v);
    }
    float max_val = _mm512_reduce_max_ps(v_max);
    for (; i < count; ++i) {
        max_val = std::max(max_val, in_logits[i]);
    }

    __m512 v_m = _mm512_set1_ps(max_val);
    __m512 v_alpha = _mm512_set1_ps(alpha);
    __m512 v_one = _mm512_set1_ps(1.0f);
    __m512 v_zero = _mm512_setzero_ps();

    // 2. Evaluate phi and compute sum
    __m512 v_sum_phi = _mm512_setzero_ps();
    i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 v = _mm512_loadu_ps(in_logits + i);
        __m512 diff = _mm512_sub_ps(v, v_m);
        __m512 phi = _mm512_max_ps(v_zero, _mm512_fmadd_ps(v_alpha, diff, v_one));
        _mm512_storeu_ps(out_probs + i, phi);
        v_sum_phi = _mm512_add_ps(v_sum_phi, phi);
    }

    float sum_phi = _mm512_reduce_add_ps(v_sum_phi);
    for (; i < count; ++i) {
        float phi = std::max(0.0f, 1.0f + alpha * (in_logits[i] - max_val));
        out_probs[i] = phi;
        sum_phi += phi;
    }

    // 3. Normalize
    float inv_sum = (sum_phi > 1e-7f) ? (1.0f / sum_phi) : 1.0f;
    __m512 v_inv = _mm512_set1_ps(inv_sum);
    i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 phi = _mm512_loadu_ps(out_probs + i);
        _mm512_storeu_ps(out_probs + i, _mm512_mul_ps(phi, v_inv));
    }
    for (; i < count; ++i) {
        out_probs[i] *= inv_sum;
    }
}

} // namespace weft
