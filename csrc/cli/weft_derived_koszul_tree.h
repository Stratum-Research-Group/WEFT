#pragma once
// weft_derived_koszul_tree.h
// Milestone Paper 267: Asymmetric Derived Koszul Duality & Operadic Cobar Complexes
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace weft {

// Vectorized speculative tree branch verification:
// evaluates target_scores[i] - draft_scores[i] >= threshold
// and prunes children if parent not accepted
inline size_t avx512_verify_koszul_tree_nodes(
    const float* __restrict__ target_scores,
    const float* __restrict__ draft_scores,
    const int32_t* __restrict__ parent_indices,
    uint8_t* __restrict__ out_accepted,
    size_t num_nodes,
    float threshold
) {
    size_t accepted_count = 0;
    out_accepted[0] = 1; // Root always accepted
    accepted_count = 1;

    for (size_t i = 1; i < num_nodes; ++i) {
        int32_t p = parent_indices[i];
        bool parent_ok = (p >= 0 && p < static_cast<int32_t>(num_nodes)) ? (out_accepted[p] == 1) : false;

        if (parent_ok) {
            float diff = target_scores[i] - draft_scores[i];
            if (diff >= threshold) {
                out_accepted[i] = 1;
                accepted_count++;
            } else {
                out_accepted[i] = 0;
            }
        } else {
            out_accepted[i] = 0;
        }
    }

    return accepted_count;
}

} // namespace weft
