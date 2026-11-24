#pragma once
// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Asymmetric p-Adic Uniformization of Shimura Curves & Čerednik-Drinfeld Theorems
// Paper 208 Native Kernel Implementation

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <immintrin.h>

namespace weft {
namespace cerednik_drinfeld {

struct TreeVertex {
    uint32_t level;
    uint32_t component;
};

// Geodesic distance on the Bruhat-Tits tree
inline uint32_t compute_tree_distance(const TreeVertex& u, const TreeVertex& v) {
    uint32_t l_diff = (u.level >= v.level) ? (u.level - v.level) : (v.level - u.level);
    uint32_t comp_penalty = (u.component == v.component) ? 0 : 2;
    return l_diff + comp_penalty;
}

// Rigid analytic codebook nearest-neighbor search along Bruhat-Tits geodesics
inline size_t find_nearest_codebook_entry(
    const TreeVertex& query,
    const std::vector<TreeVertex>& codebook)
{
    if (codebook.empty()) return 0;
    size_t best_idx = 0;
    uint32_t min_dist = compute_tree_distance(query, codebook[0]);

    for (size_t i = 1; i < codebook.size(); ++i) {
        uint32_t dist = compute_tree_distance(query, codebook[i]);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

// SIMD-accelerated batch distance evaluator
inline void batch_tree_distance_avx512(
    uint32_t q_level,
    uint32_t q_comp,
    const uint32_t* cb_levels,
    const uint32_t* cb_comps,
    uint32_t* out_dists,
    size_t count)
{
    size_t i = 0;
#if defined(__AVX512F__)
    __m512i v_q_level = _mm512_set1_epi32(static_cast<int>(q_level));
    __m512i v_q_comp  = _mm512_set1_epi32(static_cast<int>(q_comp));
    __m512i v_two     = _mm512_set1_epi32(2);

    for (; i + 16 <= count; i += 16) {
        __m512i v_cb_level = _mm512_loadu_si512((const __m512i*)(cb_levels + i));
        __m512i v_cb_comp  = _mm512_loadu_si512((const __m512i*)(cb_comps + i));

        // Absolute difference between levels
        __m512i v_diff1 = _mm512_sub_epi32(v_q_level, v_cb_level);
        __m512i v_diff2 = _mm512_sub_epi32(v_cb_level, v_q_level);
        __m512i v_ldiff = _mm512_max_epi32(v_diff1, v_diff2);

        // Component comparison
        __mmask16 comp_mask = _mm512_cmpeq_epi32_mask(v_q_comp, v_cb_comp);
        __m512i v_penalty = _mm512_mask_blend_epi32(comp_mask, v_two, _mm512_setzero_si512());

        __m512i v_res = _mm512_add_epi32(v_ldiff, v_penalty);
        _mm512_storeu_si512((__m512i*)(out_dists + i), v_res);
    }
#endif
    for (; i < count; ++i) {
        uint32_t l_diff = (q_level >= cb_levels[i]) ? (q_level - cb_levels[i]) : (cb_levels[i] - q_level);
        uint32_t comp_penalty = (q_comp == cb_comps[i]) ? 0 : 2;
        out_dists[i] = l_diff + comp_penalty;
    }
}

} // namespace cerednik_drinfeld
} // namespace weft
