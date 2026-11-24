#pragma once
// weft_motivic_singular_kv_compaction.h
// Milestone Paper 242: Native AVX-512 Motivic Singular KV Compaction
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <vector>

namespace weft {

// Vectorized token compaction based on persistence scores:
// writes indices of retained tokens (persistence >= theta) to out_retained_indices.
// Returns total count of retained tokens.
inline size_t avx512_compact_kv_by_persistence(
    const float* __restrict__ persistence_scores,
    size_t count_tokens,
    float theta_persist,
    size_t* __restrict__ out_retained_indices
) {
    __m512 v_theta = _mm512_set1_ps(theta_persist);
    size_t out_idx = 0;

    size_t i = 0;
    for (; i + 16 <= count_tokens; i += 16) {
        __m512 vp = _mm512_loadu_ps(persistence_scores + i);
        __mmask16 mask = _mm512_cmp_ps_mask(vp, v_theta, _CMP_GE_OQ);

        for (int b = 0; b < 16; ++b) {
            if (mask & (1U << b)) {
                out_retained_indices[out_idx++] = i + b;
            }
        }
    }

    for (; i < count_tokens; ++i) {
        if (persistence_scores[i] >= theta_persist) {
            out_retained_indices[out_idx++] = i;
        }
    }

    return out_idx;
}

} // namespace weft
