// weft_speculative_dag.h
// AVX-512 Vectorized Dynamic Speculative DAG Path Pruning and Threshold Filtering.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cassert>

namespace weft {

struct DAGCandidateNode {
    int token_id;
    int parent_idx;
    float cumulative_prob;
};

// Filter candidate draft nodes by threshold tau using AVX-512 SIMD masks
inline int filter_speculative_dag_nodes_avx512(
    const float* cumulative_probs,
    const int* token_ids,
    int count,
    float tau,
    int* out_surviving_ids,
    float* out_surviving_probs)
{
    __m512 v_tau = _mm512_set1_ps(tau);
    int surviving_count = 0;
    int i = 0;

    for (; i + 16 <= count; i += 16) {
        __m512 vp = _mm512_loadu_ps(&cumulative_probs[i]);
        __m512i vt = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&token_ids[i]));

        // Compare vp >= v_tau
        __mmask16 mask = _mm512_cmp_ps_mask(vp, v_tau, _CMP_GE_OQ);

        // Compress and store surviving tokens
        _mm512_mask_compressstoreu_epi32(&out_surviving_ids[surviving_count], mask, vt);
        _mm512_mask_compressstoreu_ps(&out_surviving_probs[surviving_count], mask, vp);

        surviving_count += _mm_popcnt_u32(static_cast<uint32_t>(mask));
    }

    for (; i < count; ++i) {
        if (cumulative_probs[i] >= tau) {
            out_surviving_ids[surviving_count] = token_ids[i];
            out_surviving_probs[surviving_count] = cumulative_probs[i];
            surviving_count++;
        }
    }

    return surviving_count;
}

// Target verification acceptance check:
// Accepts if uniform_rand < min(1.0, p_target / q_draft)
inline bool evaluate_target_acceptance(float p_target, float q_draft, float uniform_rand) {
    if (q_draft <= 0.0f) return true;
    float ratio = p_target / q_draft;
    float accept_prob = (ratio < 1.0f) ? ratio : 1.0f;
    return uniform_rand <= accept_prob;
}

} // namespace weft
