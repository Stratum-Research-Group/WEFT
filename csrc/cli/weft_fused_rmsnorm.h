#pragma once
// weft_fused_rmsnorm.h
// Asymmetric In-Register LayerNorm & RMSNorm Kernel Fusion (Item 5)
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized RMSNorm:
// y[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
inline void avx512_fused_rmsnorm(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    size_t dim,
    float eps = 1e-6f
) {
    __m512 v_sq_sum = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        v_sq_sum = _mm512_fmadd_ps(vx, vx, v_sq_sum);
    }

    float sq_sum = _mm512_reduce_add_ps(v_sq_sum);
    for (; i < dim; ++i) {
        sq_sum += x[i] * x[i];
    }

    float inv_rms = 1.0f / std::sqrt((sq_sum / static_cast<float>(dim)) + eps);
    __m512 v_inv_rms = _mm512_set1_ps(inv_rms);

    i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vw = _mm512_loadu_ps(weight + i);
        __m512 vy = _mm512_mul_ps(_mm512_mul_ps(vx, v_inv_rms), vw);
        _mm512_storeu_ps(y + i, vy);
    }

    for (; i < dim; ++i) {
        y[i] = (x[i] * inv_rms) * weight[i];
    }
}

} // namespace weft
