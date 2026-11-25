// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0
//
// AVX-512 Native Fused RMSNorm with Dual Residual Addition (Item 5)
// Performs out = RMSNorm(x + res1 + res2) in a single in-register pass.

#ifndef WEFT_FUSED_RMSNORM_DUAL_RESIDUAL_H
#define WEFT_FUSED_RMSNORM_DUAL_RESIDUAL_H

#include <immintrin.h>
#include <cstddef>
#include <cmath>

namespace weft {

class FusedRMSNormDualResidualEngine {
public:
    // Compute out = RMSNorm(x + res1 + res2) using gamma scale
    static void evaluate_norm(
        const float* __restrict__ x,
        const float* __restrict__ res1,
        const float* __restrict__ res2,
        const float* __restrict__ gamma,
        float* __restrict__ out,
        size_t dim,
        float epsilon = 1e-6f
    ) {
        __m512 v_sq_sum = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 vr1 = _mm512_loadu_ps(res1 + i);
            __m512 vr2 = _mm512_loadu_ps(res2 + i);
            __m512 v_sum = _mm512_add_ps(vx, _mm512_add_ps(vr1, vr2));
            v_sq_sum = _mm512_fmadd_ps(v_sum, v_sum, v_sq_sum);
        }
        float sq_sum = _mm512_reduce_add_ps(v_sq_sum);
        for (; i < dim; ++i) {
            float s = x[i] + res1[i] + res2[i];
            sq_sum += s * s;
        }

        float rms = 1.0f / std::sqrt((sq_sum / (float)dim) + epsilon);
        __m512 v_rms = _mm512_set1_ps(rms);

        i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 vr1 = _mm512_loadu_ps(res1 + i);
            __m512 vr2 = _mm512_loadu_ps(res2 + i);
            __m512 v_sum = _mm512_add_ps(vx, _mm512_add_ps(vr1, vr2));
            __m512 vg = _mm512_loadu_ps(gamma + i);
            __m512 vo = _mm512_mul_ps(_mm512_mul_ps(v_sum, v_rms), vg);
            _mm512_storeu_ps(out + i, vo);
        }
        for (; i < dim; ++i) {
            float s = x[i] + res1[i] + res2[i];
            out[i] = (s * rms) * gamma[i];
        }
    }
};

} // namespace weft

#endif // WEFT_FUSED_RMSNORM_DUAL_RESIDUAL_H
