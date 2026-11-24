#pragma once
// weft_berkovich_eviction.h
// Milestone Paper 235: Native AVX-512 Berkovich Skeletal KV Eviction
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <immintrin.h>
#include <vector>

namespace weft {

struct BerkovichTokenMetadata {
    float core_dist;
    uint32_t is_anchor; // 1 = anchor, 0 = leaf
};

// Evaluate eviction masks using AVX-512:
// evict[i] = (is_anchor[i] == 0) && (core_dist[i] > tau)
inline void avx512_berkovich_evict_mask(
    const float* __restrict__ core_dist,
    const uint32_t* __restrict__ is_anchor,
    uint8_t* __restrict__ out_evict_flags,
    size_t count,
    float tau
) {
    __m512 v_tau = _mm512_set1_ps(tau);
    __m512i v_zero_int = _mm512_setzero_si512();

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 v_dist = _mm512_loadu_ps(core_dist + i);
        __m512i v_anchor = _mm512_loadu_si512((const __m512i*)(is_anchor + i));

        // Mask where dist > tau
        __mmask16 m_dist = _mm512_cmp_ps_mask(v_dist, v_tau, _CMP_GT_OQ);
        // Mask where is_anchor == 0
        __mmask16 m_not_anchor = _mm512_cmpeq_epi32_mask(v_anchor, v_zero_int);

        // Combined eviction mask
        __mmask16 m_evict = m_dist & m_not_anchor;

        for (int b = 0; b < 16; ++b) {
            out_evict_flags[i + b] = (m_evict & (1U << b)) ? 1 : 0;
        }
    }

    for (; i < count; ++i) {
        out_evict_flags[i] = (is_anchor[i] == 0 && core_dist[i] > tau) ? 1 : 0;
    }
}

} // namespace weft
