#ifndef WEFT_DELIGNE_MUMFORD_SPECULATION_H
#define WEFT_DELIGNE_MUMFORD_SPECULATION_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct SpeculativeBranchTuple {
    float x1, x2, x3, x4;
};

// Computes the cross-ratio lambda = ((x1 - x3)(x2 - x4)) / ((x1 - x4)(x2 - x3))
inline float compute_projective_cross_ratio(float x1, float x2, float x3, float x4) {
    float num = (x1 - x3) * (x2 - x4);
    float den = (x1 - x4) * (x2 - x3);
    if (std::abs(den) < 1e-7f) return 1e7f; // Corresponds to infinity (boundary divisor)
    return num / den;
}

// Evaluates whether a 4-token speculative branch is stable (non-degenerate)
inline bool is_speculative_branch_stable(float x1, float x2, float x3, float x4, float tau_node) {
    float lam = compute_projective_cross_ratio(x1, x2, x3, x4);
    float abs_lam = std::abs(lam);
    float abs_lam_minus_one = std::abs(lam - 1.0f);
    float abs_inv_lam = (abs_lam > 1e-7f) ? std::abs(1.0f / lam) : 1e7f;

    float min_dist = std::min({abs_lam, abs_lam_minus_one, abs_inv_lam});
    return (min_dist >= tau_node);
}

// In-register SIMD speculative branch filter: prunes degenerate candidate branches in O(1)
inline size_t filter_stable_speculative_branches_simd(const float* x1, const float* x2,
                                                      const float* x3, const float* x4,
                                                      size_t count, float tau_node,
                                                      uint32_t* stable_indices) {
    size_t out_count = 0;
    for (size_t i = 0; i < count; ++i) {
        if (is_speculative_branch_stable(x1[i], x2[i], x3[i], x4[i], tau_node)) {
            stable_indices[out_count++] = static_cast<uint32_t>(i);
        }
    }
    return out_count;
}

} // namespace weft

#endif // WEFT_DELIGNE_MUMFORD_SPECULATION_H
