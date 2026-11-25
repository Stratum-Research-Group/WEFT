// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0
//
// Milestone Paper 286: Asymmetric Tropical Toric Boundary Divisors & Logarithmic Geometry
// for Non-Linear Activation Linearization.
//
// AVX-512 SIMD kernel for piecewise linear activation linearization.

#ifndef WEFT_TROPICAL_TORIC_DIVISORS_H
#define WEFT_TROPICAL_TORIC_DIVISORS_H

#include <immintrin.h>
#include <cstddef>
#include <cmath>

namespace weft {

class TropicalPiecewiseActivationEngine {
public:
    // Evaluate 3-segment piecewise linear activation:
    // x < -2 -> 0; -2 <= x < 2 -> 0.25 * (x + 2); x >= 2 -> x
    static void evaluate_pwl_activation(
        const float* __restrict__ x,
        float* __restrict__ out,
        size_t dim
    ) {
        __m512 v_two = _mm512_set1_ps(2.0f);
        __m512 v_neg_two = _mm512_set1_ps(-2.0f);
        __m512 v_quarter = _mm512_set1_ps(0.25f);
        __m512 v_zero = _mm512_setzero_ps();

        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);

            __mmask16 mask_high = _mm512_cmp_ps_mask(vx, v_two, _CMP_GE_OQ);
            __mmask16 mask_mid = _mm512_mask_cmp_ps_mask(
                _knot_mask16(mask_high), vx, v_neg_two, _CMP_GE_OQ
            );

            __m512 v_mid = _mm512_mul_ps(_mm512_add_ps(vx, v_two), v_quarter);
            __m512 v_out = _mm512_mask_blend_ps(mask_mid, v_zero, v_mid);
            v_out = _mm512_mask_blend_ps(mask_high, v_out, vx);

            _mm512_storeu_ps(out + i, v_out);
        }
        for (; i < dim; ++i) {
            float val = x[i];
            if (val >= 2.0f) out[i] = val;
            else if (val >= -2.0f) out[i] = 0.25f * (val + 2.0f);
            else out[i] = 0.0f;
        }
    }
};

} // namespace weft

#endif // WEFT_TROPICAL_TORIC_DIVISORS_H
