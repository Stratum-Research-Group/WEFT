#ifndef WEFT_SHIMURA_ROUTING_H
#define WEFT_SHIMURA_ROUTING_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <immintrin.h>

namespace weft {

struct ShimuraPointResult {
    size_t closest_cm_index;
    float killing_distance;
    float canonical_reflex_trace;
};

// Evaluates distance to canonical CM special points and routes weight vectors to minimal-drift codebooks
inline ShimuraPointResult route_to_shimura_special_point(
    const float* weight_vector,
    const float* cm_codebook, // num_points * dim
    size_t num_points,
    size_t dim
) {
    ShimuraPointResult res;
    res.closest_cm_index = 0;
    res.killing_distance = std::numeric_limits<float>::max();
    res.canonical_reflex_trace = 0.0f;

    for (size_t p = 0; p < num_points; ++p) {
        const float* point_ptr = cm_codebook + p * dim;
        float dist_sq = 0.0f;
        float dot_prod = 0.0f;

        for (size_t d = 0; d < dim; ++d) {
            float diff = weight_vector[d] - point_ptr[d];
            dist_sq += diff * diff;
            dot_prod += weight_vector[d] * point_ptr[d];
        }

        if (dist_sq < res.killing_distance) {
            res.killing_distance = dist_sq;
            res.closest_cm_index = p;
            res.canonical_reflex_trace = dot_prod;
        }
    }

    res.killing_distance = std::sqrt(res.killing_distance);
    return res;
}

} // namespace weft

#endif // WEFT_SHIMURA_ROUTING_H
