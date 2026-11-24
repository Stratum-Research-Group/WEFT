#pragma once
// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Asymmetric Fargues-Fontaine Curves on Perfectoid Fields for Multi-Token Speculative Memory Bundles
// Paper 221 Native Kernel Implementation

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <immintrin.h>

namespace weft {
namespace fargues_fontaine_tree {

struct SpeculativeBranch {
    float degree; // logit sum
    float rank;   // tree depth
};

// Filter speculative branches based on Harder-Narasimhan slope: mu = degree / rank >= tau
inline size_t filter_speculative_branches_avx512(
    const float* degrees,
    const float* ranks,
    size_t count,
    float min_slope,
    uint32_t* retained_indices)
{
    size_t retained = 0;
    size_t i = 0;

#if defined(__AVX512F__)
    __m512 v_min_slope = _mm512_set1_ps(min_slope);

    for (; i + 16 <= count; i += 16) {
        __m512 deg = _mm512_loadu_ps(degrees + i);
        __m512 rnk = _mm512_loadu_ps(ranks + i);
        __m512 slope = _mm512_div_ps(deg, rnk);

        __mmask16 mask = _mm512_cmp_ps_mask(slope, v_min_slope, _CMP_GE_OQ);

        for (int bit = 0; bit < 16; ++bit) {
            if ((mask >> bit) & 1) {
                retained_indices[retained++] = static_cast<uint32_t>(i + bit);
            }
        }
    }
#endif
    for (; i < count; ++i) {
        float slope = degrees[i] / ranks[i];
        if (slope >= min_slope) {
            retained_indices[retained++] = static_cast<uint32_t>(i);
        }
    }
    return retained;
}

} // namespace fargues_fontaine_tree
} // namespace weft
