#ifndef WEFT_TROPICAL_MODULI_STABLE_CURVES_H
#define WEFT_TROPICAL_MODULI_STABLE_CURVES_H

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace weft {

// AVX-512 Asymmetric Tropical Moduli Spaces of Stable Curves Head Router
// Computes tropical metric distances between attention heads and produces an active head bitmask.
inline uint32_t tropical_moduli_route_heads_avx512(
    const float* __restrict__ head_queries, // Shape: [num_heads, head_dim]
    size_t num_heads,
    size_t head_dim,
    float threshold = 0.05f)
{
    if (num_heads == 0 || num_heads > 32) return 0xFFFFFFFF; // Supports up to 32 heads per bitmask

    uint32_t active_mask = 0;
    // Head 0 is always active (primary root of metric tree)
    active_mask |= (1u << 0);

    __m512 vthresh = _mm512_set1_ps(threshold);

    for (size_t h = 1; h < num_heads; ++h) {
        const float* q_h = head_queries + h * head_dim;
        bool is_redundant = false;

        // Compare against already active heads
        for (size_t parent = 0; parent < h; ++parent) {
            if (!(active_mask & (1u << parent))) continue;

            const float* q_p = head_queries + parent * head_dim;
            __m512 vmin_dist = _mm512_set1_ps(1e9f);

            size_t d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 vh = _mm512_loadu_ps(q_h + d);
                __m512 vp = _mm512_loadu_ps(q_p + d);
                __m512 vdiff = _mm512_abs_ps(_mm512_sub_ps(vh, vp));
                vmin_dist = _mm512_min_ps(vmin_dist, vdiff);
            }

            float min_val = _mm512_reduce_min_ps(vmin_dist);
            for (; d < head_dim; ++d) {
                float diff = std::abs(q_h[d] - q_p[d]);
                if (diff < min_val) min_val = diff;
            }

            if (min_val < threshold) {
                is_redundant = true;
                break;
            }
        }

        if (!is_redundant) {
            active_mask |= (1u << h);
        }
    }

    return active_mask;
}

} // namespace weft

#endif // WEFT_TROPICAL_MODULI_STABLE_CURVES_H
