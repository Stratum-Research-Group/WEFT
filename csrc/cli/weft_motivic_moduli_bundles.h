#ifndef WEFT_MOTIVIC_MODULI_BUNDLES_H
#define WEFT_MOTIVIC_MODULI_BUNDLES_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct HarderNarasimhanSlope {
    float deg;
    float rk;
};

// Computes the HN slope: mu = deg / rk
inline float compute_hn_slope(float deg, float rk) {
    if (std::abs(rk) < 1e-7f) rk = 1e-7f;
    return deg / rk;
}

// In-register SIMD pruning of unstable attention heads
inline size_t prune_unstable_attention_heads_simd(const float* head_weights, const float* deg_max,
                                                  const float* rk_max, const float* deg_min,
                                                  const float* rk_min, size_t num_heads,
                                                  size_t head_dim, float tau_hn,
                                                  float* retained_heads) {
    size_t out_heads = 0;
    for (size_t h = 0; h < num_heads; ++h) {
        float mu_max = compute_hn_slope(deg_max[h], rk_max[h]);
        float mu_min = compute_hn_slope(deg_min[h], rk_min[h]);
        float slope_diff = mu_max - mu_min;

        if (slope_diff <= tau_hn) {
            // Retain semistable head
            const float* src = &head_weights[h * head_dim];
            float* dst = &retained_heads[out_heads * head_dim];
            size_t d = 0;
#if defined(__AVX2__)
            for (; d + 8 <= head_dim; d += 8) {
                __m256 v = _mm256_loadu_ps(&src[d]);
                _mm256_storeu_ps(&dst[d], v);
            }
#endif
            for (; d < head_dim; ++d) {
                dst[d] = src[d];
            }
            out_heads++;
        }
    }
    return out_heads;
}

} // namespace weft

#endif // WEFT_MOTIVIC_MODULI_BUNDLES_H
