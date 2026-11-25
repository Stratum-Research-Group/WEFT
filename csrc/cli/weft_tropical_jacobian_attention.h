#pragma once
// weft_tropical_jacobian_attention.h
// Milestone Paper 271: Asymmetric Tropical Curve Metric Graphs & Jacobian Varieties
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized tropical Abel-Jacobi distance evaluation:
// out_dists[i] = |vision[i] - text[i]|
inline void avx512_tropical_jacobian_distance(
    const float* __restrict__ vision_coords,
    const float* __restrict__ text_coords,
    float* __restrict__ out_dists,
    size_t count
) {
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vv = _mm512_loadu_ps(vision_coords + i);
        __m512 vt = _mm512_loadu_ps(text_coords + i);
        __m512 v_diff = _mm512_abs_ps(_mm512_sub_ps(vv, vt));
        _mm512_storeu_ps(out_dists + i, v_diff);
    }

    for (; i < count; ++i) {
        out_dists[i] = std::fabs(vision_coords[i] - text_coords[i]);
    }
}

} // namespace weft
