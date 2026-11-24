#pragma once
// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Asymmetric Tropical Toric Varieties & Amoeba Projections for Memory-Constrained Sparsity Bounds
// Paper 211 Native Kernel Implementation

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <immintrin.h>

namespace weft {
namespace tropical_amoeba {

struct TropicalCoordinate {
    float val1;
    float val2;
};

// Compute Chebyshev tropical norm: max(|val1|, |val2|)
inline float compute_tropical_norm(const TropicalCoordinate& p) {
    return std::max(std::abs(p.val1), std::abs(p.val2));
}

// Fast SIMD filter that discards tokens whose tropical norm exceeds threshold tau
inline size_t filter_tokens_amoeba_avx512(
    const float* val1,
    const float* val2,
    size_t count,
    float tau,
    uint32_t* retained_indices)
{
    size_t retained = 0;
    size_t i = 0;

#if defined(__AVX512F__)
    __m512 v_tau = _mm512_set1_ps(tau);
    __m512 v_zero = _mm512_setzero_ps();

    for (; i + 16 <= count; i += 16) {
        __m512 v1 = _mm512_loadu_ps(val1 + i);
        __m512 v2 = _mm512_loadu_ps(val2 + i);

        // abs using max(v, -v) with AVX512F
        __m512 neg_v1 = _mm512_sub_ps(v_zero, v1);
        __m512 neg_v2 = _mm512_sub_ps(v_zero, v2);
        __m512 abs_v1 = _mm512_max_ps(v1, neg_v1);
        __m512 abs_v2 = _mm512_max_ps(v2, neg_v2);

        __m512 max_v = _mm512_max_ps(abs_v1, abs_v2);
        __mmask16 mask = _mm512_cmple_ps_mask(max_v, v_tau);

        for (int bit = 0; bit < 16; ++bit) {
            if ((mask >> bit) & 1) {
                retained_indices[retained++] = static_cast<uint32_t>(i + bit);
            }
        }
    }
#endif
    for (; i < count; ++i) {
        float norm = std::max(std::abs(val1[i]), std::abs(val2[i]));
        if (norm <= tau) {
            retained_indices[retained++] = static_cast<uint32_t>(i);
        }
    }
    return retained;
}

} // namespace tropical_amoeba
} // namespace weft
