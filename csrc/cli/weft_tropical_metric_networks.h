#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

namespace weft {

// Tropical Kleene star distance closure for P2P edge attention routing
// Solves (A (x) B)_ij = min_k (A_ik + B_kj) via AVX-512 SIMD
class TropicalMetricRouter {
public:
    static void compute_tropical_closure(const float* adj_matrix, float* dist_matrix, size_t n) {
        if (!adj_matrix || !dist_matrix || n == 0) return;

        // Initialize dist_matrix with adj_matrix
        for (size_t i = 0; i < n * n; ++i) {
            dist_matrix[i] = adj_matrix[i];
        }

        // Tropical Floyd-Warshall / Kleene iteration: dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
        for (size_t k = 0; k < n; ++k) {
            for (size_t i = 0; i < n; ++i) {
                float dik = dist_matrix[i * n + k];
                __m512 dik_vec = _mm512_set1_ps(dik);

                size_t j = 0;
                for (; j + 16 <= n; j += 16) {
                    __m512 dij_vec = _mm512_loadu_ps(&dist_matrix[i * n + j]);
                    __m512 dkj_vec = _mm512_loadu_ps(&dist_matrix[k * n + j]);
                    __m512 sum_vec = _mm512_add_ps(dik_vec, dkj_vec);
                    __m512 min_vec = _mm512_min_ps(dij_vec, sum_vec);
                    _mm512_storeu_ps(&dist_matrix[i * n + j], min_vec);
                }
                for (; j < n; ++j) {
                    float sum = dik + dist_matrix[k * n + j];
                    dist_matrix[i * n + j] = std::min(dist_matrix[i * n + j], sum);
                }
            }
        }
    }

    // Selects optimal peer destination for KV shard transfer given source node and candidate peers
    static size_t select_best_peer(const float* dist_matrix, size_t n, size_t src, const size_t* candidates, size_t num_candidates) {
        if (!dist_matrix || !candidates || num_candidates == 0) return src;

        size_t best_peer = candidates[0];
        float min_cost = dist_matrix[src * n + candidates[0]];

        for (size_t i = 1; i < num_candidates; ++i) {
            float cost = dist_matrix[src * n + candidates[i]];
            if (cost < min_cost) {
                min_cost = cost;
                best_peer = candidates[i];
            }
        }
        return best_peer;
    }
};

} // namespace weft
