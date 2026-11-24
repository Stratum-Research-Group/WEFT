// weft_tropical_polytope.h
// Native AVX-512 in-register Tropical Polytope Convex Hull Branch Bounding.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <cassert>

namespace weft {

// Evaluate whether the candidate branch vectors lie strictly outside acceptance threshold tau:
// Branch is pruned if min_{v in V} min_{j} (weights[j] + v[j]) > tau
inline bool should_prune_tropical_branch_avx512(const float* candidate_matrix, // rows: num_candidates, cols: dim
                                                const float* target_weights,
                                                int num_candidates,
                                                int dim,
                                                float tau) {
    float global_min = std::numeric_limits<float>::infinity();

    for (int r = 0; r < num_candidates; ++r) {
        const float* cand = &candidate_matrix[r * dim];
        __m512 v_min = _mm512_set1_ps(std::numeric_limits<float>::infinity());

        int c = 0;
        for (; c + 16 <= dim; c += 16) {
            __m512 vw = _mm512_loadu_ps(&target_weights[c]);
            __m512 vc = _mm512_loadu_ps(&cand[c]);
            __m512 vsum = _mm512_add_ps(vw, vc);
            v_min = _mm512_min_ps(v_min, vsum);
        }

        float cand_min = _mm512_reduce_min_ps(v_min);
        for (; c < dim; ++c) {
            cand_min = std::min(cand_min, target_weights[c] + cand[c]);
        }
        global_min = std::min(global_min, cand_min);
    }

    return global_min > tau;
}

// Compute tropical convex combination of two vectors: z_j = min(alpha + x_j, beta + y_j)
inline void tropical_convex_combination(const float* x, const float* y, float* z, int dim, float alpha, float beta) {
    for (int j = 0; j < dim; ++j) {
        z[j] = std::min(alpha + x[j], beta + y[j]);
    }
}

} // namespace weft
