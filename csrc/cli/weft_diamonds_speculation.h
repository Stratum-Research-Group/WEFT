#ifndef WEFT_DIAMONDS_SPECULATION_H
#define WEFT_DIAMONDS_SPECULATION_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct DiamondBranchProfile {
    size_t num_candidates;
    size_t accepted_candidates;
    float max_cocycle_residual;
    bool is_v_descent_satisfied;
};

// Evaluates v-descent cocycle conditions across speculative diamond tree branches
// p_1^* s == p_2^* s on fiber product
inline DiamondBranchProfile verify_diamond_speculative_branches(
    const float* draft_projections,
    const float* target_verifications,
    size_t num_candidates,
    size_t feature_dim,
    float cocycle_threshold
) {
    DiamondBranchProfile prof;
    prof.num_candidates = num_candidates;
    prof.accepted_candidates = 0;
    prof.max_cocycle_residual = 0.0f;

    float max_res = 0.0f;

    for (size_t c = 0; c < num_candidates; ++c) {
        const float* d_ptr = draft_projections + c * feature_dim;
        const float* t_ptr = target_verifications + c * feature_dim;

        float branch_diff_sq = 0.0f;
        for (size_t d = 0; d < feature_dim; ++d) {
            float diff = d_ptr[d] - t_ptr[d];
            branch_diff_sq += diff * diff;
        }

        float branch_diff = std::sqrt(branch_diff_sq);
        if (branch_diff > max_res) {
            max_res = branch_diff;
        }

        if (branch_diff <= cocycle_threshold) {
            prof.accepted_candidates++;
        }
    }

    prof.max_cocycle_residual = max_res;
    prof.is_v_descent_satisfied = (prof.accepted_candidates == num_candidates);

    return prof;
}

} // namespace weft

#endif // WEFT_DIAMONDS_SPECULATION_H
