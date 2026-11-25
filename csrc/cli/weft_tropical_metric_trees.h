#ifndef WEFT_TROPICAL_METRIC_TREES_H
#define WEFT_TROPICAL_METRIC_TREES_H

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace weft {

// AVX-512 Asymmetric Tropical Moduli of Metric Trees Geodesic Routing Evaluator
// Evaluates min-plus geodesic path lengths across 16 micro-cluster nodes in parallel.
inline void tropical_metric_trees_route_16nodes_avx512(
    const float* __restrict__ cost_matrix_16x16, // Row-major 16x16 edge latency matrix
    float* __restrict__ dist_matrix_16x16,       // Output shortest geodesic distance matrix
    uint8_t* __restrict__ next_hop_matrix_16x16) // Optimal next-hop forwarder
{
    // Initialize dist matrix from direct cost matrix
    for (size_t i = 0; i < 256; ++i) {
        dist_matrix_16x16[i] = cost_matrix_16x16[i];
    }
    for (uint8_t i = 0; i < 16; ++i) {
        for (uint8_t j = 0; j < 16; ++j) {
            next_hop_matrix_16x16[i * 16 + j] = (i == j) ? i : j;
        }
    }

    // Tropical Floyd-Warshall via AVX-512 min-plus vector operations
    for (size_t k = 0; k < 16; ++k) {
        __m512 vdist_k_row = _mm512_loadu_ps(dist_matrix_16x16 + k * 16);

        for (size_t i = 0; i < 16; ++i) {
            float dist_ik = dist_matrix_16x16[i * 16 + k];
            __m512 vdist_ik = _mm512_set1_ps(dist_ik);

            // Path through k: d(i, k) + d(k, j)
            __m512 vcandidate = _mm512_add_ps(vdist_ik, vdist_k_row);

            __m512 vcurr = _mm512_loadu_ps(dist_matrix_16x16 + i * 16);

            // Compare: candidate < current
            __mmask16 mask = _mm512_cmp_ps_mask(vcandidate, vcurr, _CMP_LT_OQ);

            if (mask) {
                __m512 vupdated = _mm512_mask_blend_ps(mask, vcurr, vcandidate);
                _mm512_storeu_ps(dist_matrix_16x16 + i * 16, vupdated);

                // Update next hop for elements in mask
                for (size_t j = 0; j < 16; ++j) {
                    if (mask & (1u << j)) {
                        next_hop_matrix_16x16[i * 16 + j] = next_hop_matrix_16x16[i * 16 + k];
                    }
                }
            }
        }
    }
}

} // namespace weft

#endif // WEFT_TROPICAL_METRIC_TREES_H
