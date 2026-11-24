#pragma once
// weft_tropical_bergman_moe.h
// Milestone Paper 256: Asymmetric Tropical Hyperplane Arrangements & Bergman Fans for Sparse MoE Expert Selection
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized tropical Bergman distance to spine:
// out_distance[i] = |valuations[i] - max_val|
inline void avx512_tropical_bergman_routing(
    const float* __restrict__ valuations,
    float* __restrict__ out_distance,
    size_t num_experts,
    float max_val
) {
    __m512 v_max = _mm512_set1_ps(max_val);

    size_t i = 0;
    for (; i + 16 <= num_experts; i += 16) {
        __m512 v_val = _mm512_loadu_ps(valuations + i);
        __m512 v_diff = _mm512_sub_ps(v_val, v_max);
        __m512 v_abs = _mm512_abs_ps(v_diff);
        _mm512_storeu_ps(out_distance + i, v_abs);
    }

    for (; i < num_experts; ++i) {
        out_distance[i] = std::fabs(valuations[i] - max_val);
    }
}

} // namespace weft
