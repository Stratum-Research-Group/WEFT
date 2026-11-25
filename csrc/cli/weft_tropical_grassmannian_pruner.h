#ifndef WEFT_TROPICAL_GRASSMANNIAN_PRUNER_H
#define WEFT_TROPICAL_GRASSMANNIAN_PRUNER_H

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace weft {

// AVX-512 Asymmetric Tropical Grassmannian Dynamic Sparsification Kernel
// Filters out activation channels whose magnitude falls below the tropical matroid threshold.
// Writes sparsified vector and outputs the count of non-zero channels.
inline size_t tropical_grassmannian_prune_avx512(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t dim,
    float threshold = 0.05f)
{
    __m512 vthresh = _mm512_set1_ps(threshold);
    __m512 vzero = _mm512_setzero_ps();
    size_t active_count = 0;

    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 v = _mm512_loadu_ps(input + i);
        __m512 vabs = _mm512_abs_ps(v);

        // Keep channel if |v| >= threshold
        __mmask16 keep_mask = _mm512_cmp_ps_mask(vabs, vthresh, _CMP_GE_OQ);

        // Blend with zero
        __m512 vpruned = _mm512_mask_blend_ps(keep_mask, vzero, v);
        _mm512_storeu_ps(output + i, vpruned);

        active_count += _mm_popcnt_u32(static_cast<uint32_t>(keep_mask));
    }

    for (; i < dim; ++i) {
        if (std::abs(input[i]) >= threshold) {
            output[i] = input[i];
            active_count++;
        } else {
            output[i] = 0.0f;
        }
    }

    return active_count;
}

} // namespace weft

#endif // WEFT_TROPICAL_GRASSMANNIAN_PRUNER_H
