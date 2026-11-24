#pragma once
// weft_motivic_etale_quantization.h
// Milestone Paper 251: Native AVX-512 Motivic Étale Sub-Bit Quantization
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <immintrin.h>

namespace weft {

struct ScaleInvariantTernaryConfig {
    float delta_ratio; // Threshold ratio relative to mean absolute value (e.g. 0.7f)
};

// Vectorized scale-invariant ternary quantization using AVX-512:
// 1. Compute dynamic threshold = delta_ratio * mean(|x|)
// 2. out_q[i] = +1 if x >= threshold, -1 if x <= -threshold, 0 otherwise
inline void avx512_scale_invariant_ternary_quantize(
    const float* __restrict__ in_x,
    float* __restrict__ out_q,
    size_t count,
    float delta_ratio,
    float* __restrict__ out_scale
) {
    if (count == 0) return;

    // 1. Compute mean absolute value
    __m512 v_sum_abs = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vx = _mm512_loadu_ps(in_x + i);
        v_sum_abs = _mm512_add_ps(v_sum_abs, _mm512_abs_ps(vx));
    }
    float sum_abs = _mm512_reduce_add_ps(v_sum_abs);
    for (; i < count; ++i) {
        sum_abs += std::fabs(in_x[i]);
    }

    float mean_abs = sum_abs / static_cast<float>(count);
    float threshold = delta_ratio * mean_abs;
    if (out_scale) *out_scale = mean_abs;

    __m512 v_th = _mm512_set1_ps(threshold);
    __m512 v_neg_th = _mm512_set1_ps(-threshold);
    __m512 v_one = _mm512_set1_ps(1.0f);
    __m512 v_neg_one = _mm512_set1_ps(-1.0f);
    __m512 v_zero = _mm512_setzero_ps();

    // 2. Vectorized thresholding
    i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vx = _mm512_loadu_ps(in_x + i);
        __mmask16 m_pos = _mm512_cmp_ps_mask(vx, v_th, _CMP_GE_OQ);
        __mmask16 m_neg = _mm512_cmp_ps_mask(vx, v_neg_th, _CMP_LE_OQ);

        __m512 v_res = _mm512_mask_blend_ps(m_pos, v_zero, v_one);
        v_res = _mm512_mask_blend_ps(m_neg, v_res, v_neg_one);

        _mm512_storeu_ps(out_q + i, v_res);
    }

    for (; i < count; ++i) {
        if (in_x[i] >= threshold) {
            out_q[i] = 1.0f;
        } else if (in_x[i] <= -threshold) {
            out_q[i] = -1.0f;
        } else {
            out_q[i] = 0.0f;
        }
    }
}

} // namespace weft
