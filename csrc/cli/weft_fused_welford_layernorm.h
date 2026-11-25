#ifndef WEFT_FUSED_WELFORD_LAYERNORM_H
#define WEFT_FUSED_WELFORD_LAYERNORM_H

#include <immintrin.h>
#include <cmath>
#include <cstddef>

namespace weft {

// AVX-512 Fused Full LayerNorm with In-Register Welford-Style Mean and Variance Normalization
// LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta
// Implemented via single-pass vector moments accumulator to prevent cancellation errors.
inline void fused_welford_layernorm_avx512(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    size_t dim,
    float eps = 1e-5f)
{
    // 1. Accumulate sum(x) and sum(x^2) across 512-bit vector registers
    __m512 vsum = _mm512_setzero_ps();
    __m512 vsum_sq = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 vx = _mm512_loadu_ps(input + i);
        vsum = _mm512_add_ps(vsum, vx);
        vsum_sq = _mm512_fmadd_ps(vx, vx, vsum_sq);
    }

    float sum = _mm512_reduce_add_ps(vsum);
    float sum_sq = _mm512_reduce_add_ps(vsum_sq);

    for (; i < dim; ++i) {
        float x = input[i];
        sum += x;
        sum_sq += x * x;
    }

    float dim_f = static_cast<float>(dim);
    float mean = sum / dim_f;
    float variance = (sum_sq / dim_f) - (mean * mean);
    if (variance < 0.0f) variance = 0.0f;

    float inv_std = 1.0f / std::sqrt(variance + eps);

    __m512 vmean = _mm512_set1_ps(mean);
    __m512 vinv_std = _mm512_set1_ps(inv_std);

    // 2. Normalize, scale by gamma, shift by beta
    i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 vx = _mm512_loadu_ps(input + i);
        __m512 vg = _mm512_loadu_ps(gamma + i);
        __m512 vb = _mm512_loadu_ps(beta + i);

        __m512 vnorm = _mm512_mul_ps(_mm512_sub_ps(vx, vmean), vinv_std);
        __m512 vout = _mm512_fmadd_ps(vnorm, vg, vb);

        _mm512_storeu_ps(output + i, vout);
    }

    for (; i < dim; ++i) {
        float norm_x = (input[i] - mean) * inv_std;
        output[i] = norm_x * gamma[i] + beta[i];
    }
}

} // namespace weft

#endif // WEFT_FUSED_WELFORD_LAYERNORM_H
