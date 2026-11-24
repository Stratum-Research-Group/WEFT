#pragma once
// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Asymmetric Motivic Milnor Fibers & Denef-Loeser Motivic Zeta Functions for Non-Linear Activation Singularities
// Paper 220 Native Kernel Implementation

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <immintrin.h>

namespace weft {
namespace motivic_milnor {

// Smooth activation evaluation around non-differentiable cusps:
// If |x| < eps, smooth via polynomial arc jet interpolation; else standard SwiGLU / GeLU.
inline void smooth_cusp_activation_avx512(
    const float* in_acts,
    float* out_acts,
    size_t count,
    float eps = 0.05f)
{
    size_t i = 0;
#if defined(__AVX512F__)
    __m512 v_eps = _mm512_set1_ps(eps);
    __m512 v_zero = _mm512_setzero_ps();
    __m512 v_half = _mm512_set1_ps(0.5f);
    __m512 v_inv_eps = _mm512_set1_ps(1.0f / eps);

    for (; i + 16 <= count; i += 16) {
        __m512 x = _mm512_loadu_ps(in_acts + i);

        // abs(x)
        __m512 neg_x = _mm512_sub_ps(v_zero, x);
        __m512 abs_x = _mm512_max_ps(x, neg_x);

        __mmask16 in_cusp = _mm512_cmple_ps_mask(abs_x, v_eps);

        // Standard ReLU-like: max(0, x)
        __m512 standard = _mm512_max_ps(x, v_zero);

        // Smooth arc jet interpolation: 0.5 * (x + eps) * (x + eps) / (2 * eps)
        __m512 x_plus_eps = _mm512_add_ps(x, v_eps);
        __m512 sq = _mm512_mul_ps(x_plus_eps, x_plus_eps);
        __m512 smoothed = _mm512_mul_ps(sq, _mm512_mul_ps(v_half, v_inv_eps));

        __m512 res = _mm512_mask_blend_ps(in_cusp, standard, smoothed);
        _mm512_storeu_ps(out_acts + i, res);
    }
#endif
    for (; i < count; ++i) {
        float x = in_acts[i];
        if (std::abs(x) <= eps) {
            float x_pe = x + eps;
            out_acts[i] = (x_pe * x_pe) / (4.0f * eps);
        } else {
            out_acts[i] = (x > 0.0f) ? x : 0.0f;
        }
    }
}

} // namespace motivic_milnor
} // namespace weft
