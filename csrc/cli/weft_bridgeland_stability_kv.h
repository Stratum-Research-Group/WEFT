#ifndef WEFT_BRIDGELAND_STABILITY_KV_H
#define WEFT_BRIDGELAND_STABILITY_KV_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct BridgelandCentralCharge {
    float deg;
    float rk;
    float phase;
};

// Computes the central charge phase phi in (0, 1]: phi = atan2(rk, -deg) / pi
inline float compute_central_charge_phase(float deg, float rk) {
    constexpr float INV_PI = 0.3183098861837907f;
    float angle = std::atan2(rk, -deg); // Range (-pi, pi]
    if (angle <= 0.0f) angle += 6.283185307179586f; // Map to (0, 2pi]
    return angle * INV_PI * 0.5f; // Normalize to (0, 1]
}

// In-register SIMD pruning of KV-cache based on Bridgeland semistable phase threshold
inline size_t compact_kv_cache_bridgeland_simd(const float* kv_in, const float* deg, const float* rk,
                                               size_t num_tokens, size_t dim,
                                               float phi_crit, float* kv_out) {
    size_t out_tokens = 0;
    for (size_t t = 0; t < num_tokens; ++t) {
        float phase = compute_central_charge_phase(deg[t], rk[t]);
        if (phase >= phi_crit) {
            // Retain semistable factor
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

#endif // WEFT_BRIDGELAND_STABILITY_KV_H
