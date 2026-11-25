#ifndef WEFT_DERIVED_INTERSECTION_HOMOLOGY_H
#define WEFT_DERIVED_INTERSECTION_HOMOLOGY_H

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <algorithm>

namespace weft {

// AVX-512 Asymmetric Derived Intersection Homology & Perverse Sheaf Normalization Kernel
// Normalizes activations over singular stratified sub-bit activation spaces
inline void perverse_sheaf_normalize_avx512(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    size_t dim,
    float eps = 1e-5f)
{
    // 1. Compute second moment (variance) across regular strata
    __m512 sum_sq = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 vx = _mm512_loadu_ps(input + i);
        sum_sq = _mm512_fmadd_ps(vx, vx, sum_sq);
    }
    float total_sq = _mm512_reduce_add_ps(sum_sq);
    for (; i < dim; ++i) {
        total_sq += input[i] * input[i];
    }

    float mean_sq = total_sq / static_cast<float>(dim);
    float inv_std = 1.0f / std::sqrt(mean_sq + eps);
    __m512 vinv_std = _mm512_set1_ps(inv_std);

    // 2. Middle perversity truncation parameters
    // Normalizing singular divergence while preserving invariant norm
    __m512 vone = _mm512_set1_ps(1.0f);
    __m512 vzero = _mm512_setzero_ps();
    __m512 valpha = _mm512_set1_ps(0.125f); // Soft perverse stratum regularization

    i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 vx = _mm512_loadu_ps(input + i);
        __m512 vg = _mm512_loadu_ps(gamma + i);

        // Normalized activation
        __m512 vnorm = _mm512_mul_ps(vx, vinv_std);

        // Perverse filter: phi(u) = clamp(1.0 - alpha * |u|, 0.0, 1.0)
        __m512 vabs = _mm512_abs_ps(vnorm);
        __m512 vfactor = _mm512_sub_ps(vone, _mm512_mul_ps(valpha, vabs));
        vfactor = _mm512_max_ps(vzero, _mm512_min_ps(vone, vfactor));

        // Combined perverse projection
        __m512 vout = _mm512_mul_ps(_mm512_mul_ps(vnorm, vfactor), vg);
        _mm512_storeu_ps(output + i, vout);
    }

    for (; i < dim; ++i) {
        float x = input[i];
        float norm_x = x * inv_std;
        float factor = std::clamp(1.0f - 0.125f * std::abs(norm_x), 0.0f, 1.0f);
        output[i] = norm_x * factor * gamma[i];
    }
}

} // namespace weft

#endif // WEFT_DERIVED_INTERSECTION_HOMOLOGY_H
