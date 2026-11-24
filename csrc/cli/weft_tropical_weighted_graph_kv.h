#pragma once
// weft_tropical_weighted_graph_kv.h
// Milestone Paper 250: Native AVX-512 Tropical Weighted Graph KV Eviction
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <vector>

namespace weft {

// Vectorized cluster bridge distance evaluation using AVX-512:
// evict[i] = (bridge_dist[i] > tau_bridge)
inline void avx512_evaluate_cluster_eviction(
    const float* __restrict__ bridge_distances,
    uint8_t* __restrict__ out_evict_flags,
    size_t count_clusters,
    float tau_bridge
) {
    __m512 v_tau = _mm512_set1_ps(tau_bridge);

    size_t i = 0;
    for (; i + 16 <= count_clusters; i += 16) {
        __m512 vd = _mm512_loadu_ps(bridge_distances + i);
        __mmask16 mask = _mm512_cmp_ps_mask(vd, v_tau, _CMP_GT_OQ);

        for (int b = 0; b < 16; ++b) {
            out_evict_flags[i + b] = (mask & (1U << b)) ? 1 : 0;
        }
    }

    for (; i < count_clusters; ++i) {
        out_evict_flags[i] = (bridge_distances[i] > tau_bridge) ? 1 : 0;
    }
}

} // namespace weft
