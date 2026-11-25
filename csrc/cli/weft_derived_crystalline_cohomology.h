#ifndef WEFT_DERIVED_CRYSTALLINE_COHOMOLOGY_H
#define WEFT_DERIVED_CRYSTALLINE_COHOMOLOGY_H

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>

namespace weft {

// AVX-512 Asymmetric Derived Crystalline Cohomology Recalibration Kernel
// Recalibrates sub-bit layer scale alpha using in-register Frobenius slope projections.
inline float crystalline_recalibrate_scale_avx512(
    const float* __restrict__ full_act,
    const int8_t* __restrict__ ternary_weights, // {-1, 0, +1}
    float current_scale,
    size_t dim,
    int prime = 3)
{
    if (dim == 0 || current_scale <= 0.0f) return current_scale;

    // 1. Accumulate projected activation and ternary weight inner products
    __m512 vsum_error = _mm512_setzero_ps();
    __m512 vsum_ternary = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 vx = _mm512_loadu_ps(full_act + i);

        // Load 16 int8 values and convert to float
        __m128i vi8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ternary_weights + i));
        __m512i vi32 = _mm512_cvtepi8_epi32(vi8);
        __m512 vw = _mm512_cvtepi32_ps(vi32);

        // Predicted activation = scale * w
        __m512 vpred = _mm512_mul_ps(_mm512_set1_ps(current_scale), vw);
        __m512 vdiff = _mm512_sub_ps(vx, vpred);

        vsum_error = _mm512_fmadd_ps(vdiff, vw, vsum_error);
        vsum_ternary = _mm512_fmadd_ps(vw, vw, vsum_ternary);
    }

    float error_proj = _mm512_reduce_add_ps(vsum_error);
    float ternary_norm = _mm512_reduce_add_ps(vsum_ternary);

    for (; i < dim; ++i) {
        float x = full_act[i];
        float w = static_cast<float>(ternary_weights[i]);
        float diff = x - current_scale * w;
        error_proj += diff * w;
        ternary_norm += w * w;
    }

    if (ternary_norm < 1e-5f) return current_scale;

    // 2. Witt Frobenius Slope Projection:
    // delta = (1 / prime) * (error_proj / ternary_norm)
    float slope_correction = (1.0f / static_cast<float>(prime)) * (error_proj / ternary_norm);
    float recalibrated_scale = current_scale + slope_correction;

    return std::max(recalibrated_scale, 1e-4f);
}

} // namespace weft

#endif // WEFT_DERIVED_CRYSTALLINE_COHOMOLOGY_H
