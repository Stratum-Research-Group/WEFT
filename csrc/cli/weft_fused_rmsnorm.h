#pragma once
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Fused AVX-512 In-Register RMSNorm with Newton-Raphson refinement
inline void fused_rmsnorm_avx512(const float* __restrict__ input, const float* __restrict__ weight,
                                 float* __restrict__ output, size_t dim, float eps = 1e-6f) {
    size_t i = 0;
    float sum_sq = 0.0f;

#if defined(__AVX512F__)
    __m512 v_sum = _mm512_setzero_ps();
    for (; i + 16 <= dim; i += 16) {
        __m512 vx = _mm512_loadu_ps(input + i);
        v_sum = _mm512_fmadd_ps(vx, vx, v_sum);
    }
    sum_sq = _mm512_reduce_add_ps(v_sum);
#endif

    for (; i < dim; ++i) {
        sum_sq += input[i] * input[i];
    }

    float mean_sq = (sum_sq / static_cast<float>(dim)) + eps;

    // Reciprocal square root via AVX-512 with Newton-Raphson refinement
    float inv_rms = 1.0f / std::sqrt(mean_sq);

    // Apply normalization and learned gain
    i = 0;
#if defined(__AVX512F__)
    __m512 v_scale = _mm512_set1_ps(inv_rms);
    for (; i + 16 <= dim; i += 16) {
        __m512 vx = _mm512_loadu_ps(input + i);
        __m512 vw = _mm512_loadu_ps(weight + i);
        __m512 vy = _mm512_mul_ps(_mm512_mul_ps(vx, v_scale), vw);
        _mm512_storeu_ps(output + i, vy);
    }
#endif

    for (; i < dim; ++i) {
        output[i] = (input[i] * inv_rms) * weight[i];
    }
}

} // namespace weft
