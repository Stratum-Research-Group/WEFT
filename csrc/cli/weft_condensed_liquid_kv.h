#ifndef WEFT_CONDENSED_LIQUID_KV_H
#define WEFT_CONDENSED_LIQUID_KV_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

// Computes the liquid p-norm for p = 0.5: sum_i sqrt(|x_i|)
inline float compute_liquid_half_norm(const float* vec, size_t dim) {
    float sum = 0.0f;
    size_t i = 0;
#if defined(__AVX2__)
    __m256 v_sum = _mm256_setzero_ps();
    __m256 v_sign_mask = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= dim; i += 8) {
        __m256 v_x = _mm256_loadu_ps(&vec[i]);
        __m256 v_abs = _mm256_andnot_ps(v_sign_mask, v_x);
        __m256 v_sqrt = _mm256_sqrt_ps(v_abs);
        v_sum = _mm256_add_ps(v_sum, v_sqrt);
    }
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, v_sum);
    for (int k = 0; k < 8; ++k) sum += tmp[k];
#endif
    for (; i < dim; ++i) {
        sum += std::sqrt(std::abs(vec[i]));
    }
    return sum;
}

// In-register SIMD pruning of KV cache retaining tokens within liquid p-norm bounds
inline size_t compact_kv_cache_liquid_simd(const float* kv_in, size_t num_tokens, size_t dim,
                                           float tau_liquid, float* kv_out) {
    size_t out_tokens = 0;
    for (size_t t = 0; t < num_tokens; ++t) {
        const float* src = &kv_in[t * dim];
        float norm_val = compute_liquid_half_norm(src, dim);
        if (norm_val >= tau_liquid) {
            float* dst = &kv_out[out_tokens * dim];
            size_t d = 0;
#if defined(__AVX2__)
            for (; d + 8 <= dim; d += 8) {
                __m256 v = _mm256_loadu_ps(&src[d]);
                _mm256_storeu_ps(&dst[d], v);
            }
#endif
            for (; d < dim; ++d) {
                dst[d] = src[d];
            }
            out_tokens++;
        }
    }
    return out_tokens;
}

// Internal helper for liquid norm evaluation
inline float compute_liquid_half_half_norm_helper(const float* vec, size_t dim) {
    return compute_liquid_half_norm(vec, dim);
}

} // namespace weft

#endif // WEFT_CONDENSED_LIQUID_KV_H
