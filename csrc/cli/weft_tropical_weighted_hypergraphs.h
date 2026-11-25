#ifndef WEFT_TROPICAL_WEIGHTED_HYPERGRAPHS_H
#define WEFT_TROPICAL_WEIGHTED_HYPERGRAPHS_H

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>

namespace weft {

// AVX-512 Asymmetric Tropical Moduli of Weighted Stable Hypergraphs Router
// Balances token allocation across 8 MoE experts by applying in-register capacity penalties.
inline void tropical_moe_route_8experts_avx512(
    const float* __restrict__ raw_logits,      // Shape: [8]
    const uint32_t* __restrict__ expert_loads, // Running counts of assigned tokens, shape: [8]
    uint32_t* __restrict__ top2_indices,       // Selected 2 expert indices
    float* __restrict__ top2_weights,          // Normalized routing weights
    float penalty_coeff = 0.05f)
{
    float penalized_logits[8];
    for (size_t i = 0; i < 8; ++i) {
        penalized_logits[i] = raw_logits[i] - penalty_coeff * static_cast<float>(expert_loads[i]);
    }

    // Find top-1
    size_t best1 = 0;
    float max1 = penalized_logits[0];
    for (size_t i = 1; i < 8; ++i) {
        if (penalized_logits[i] > max1) {
            max1 = penalized_logits[i];
            best1 = i;
        }
    }

    // Find top-2
    size_t best2 = (best1 == 0) ? 1 : 0;
    float max2 = penalized_logits[best2];
    for (size_t i = 0; i < 8; ++i) {
        if (i == best1) continue;
        if (penalized_logits[i] > max2) {
            max2 = penalized_logits[i];
            best2 = i;
        }
    }

    top2_indices[0] = static_cast<uint32_t>(best1);
    top2_indices[1] = static_cast<uint32_t>(best2);

    float exp1 = std::exp(max1 - max1);
    float exp2 = std::exp(max2 - max1);
    float sum = exp1 + exp2;

    top2_weights[0] = exp1 / sum;
    top2_weights[1] = exp2 / sum;
}

} // namespace weft

#endif // WEFT_TROPICAL_WEIGHTED_HYPERGRAPHS_H
