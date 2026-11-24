#ifndef WEFT_DONALDSON_THOMAS_KV_H
#define WEFT_DONALDSON_THOMAS_KV_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct TokenIdealSheaf {
    uint32_t id;
    float behrend_weight;
    float virtual_obstruction_norm;
    bool is_retained;
};

// Evaluates Behrend constructible function and DT weight from local curvature
inline float compute_behrend_weight(float curvature_norm, float tau_obs) {
    if (curvature_norm < tau_obs) return 0.0f;
    return std::floor(curvature_norm / tau_obs);
}

// In-register SIMD pruning of KV-cache based on Donaldson-Thomas non-vanishing criteria
inline size_t compact_kv_cache_donaldson_thomas_simd(const float* kv_in, const float* curvature,
                                                     size_t num_tokens, size_t dim,
                                                     float tau_obs, float* kv_out) {
    size_t out_tokens = 0;
    for (size_t t = 0; t < num_tokens; ++t) {
        float curv = curvature[t];
        float w = compute_behrend_weight(curv, tau_obs);
        if (w > 0.0f) {
            // Retain token sheaf
            const float* src = &kv_in[t * dim];
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

} // namespace weft

#endif // WEFT_DONALDSON_THOMAS_KV_H
