#pragma once
// weft_tropical_convexity_kv.h
// Milestone Paper 268: Asymmetric Tropical Convexity & Polytopal Core Allocations
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized tropical polytope membership distance evaluation:
// out_distances[i] = max(0, addr[i] - page_base[i] - page_size[i]) + max(0, page_base[i] - addr[i])
inline void avx512_tropical_polytope_isolation(
    const float* __restrict__ query_addrs,
    const float* __restrict__ page_base,
    const float* __restrict__ page_size,
    float* __restrict__ out_violations,
    size_t count
) {
    __m512 v_zero = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 va = _mm512_loadu_ps(query_addrs + i);
        __m512 vb = _mm512_loadu_ps(page_base + i);
        __m512 vs = _mm512_loadu_ps(page_size + i);

        __m512 v_upper = _mm512_add_ps(vb, vs);
        __m512 v_above = _mm512_max_ps(_mm512_sub_ps(va, v_upper), v_zero);
        __m512 v_below = _mm512_max_ps(_mm512_sub_ps(vb, va), v_zero);

        __m512 v_viol = _mm512_add_ps(v_above, v_below);
        _mm512_storeu_ps(out_violations + i, v_viol);
    }

    for (; i < count; ++i) {
        float above = std::max(0.0f, query_addrs[i] - (page_base[i] + page_size[i]));
        float below = std::max(0.0f, page_base[i] - query_addrs[i]);
        out_violations[i] = above + below;
    }
}

} // namespace weft
