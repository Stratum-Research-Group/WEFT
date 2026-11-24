#ifndef WEFT_UNITARY_SHIMURA_H
#define WEFT_UNITARY_SHIMURA_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <immintrin.h>

namespace weft {

struct PicardRoutingProfile {
    size_t selected_expert_id;
    float bergman_distance;
    float routing_confidence;
    bool is_inside_ball;
};

// Evaluates Bergman metric distances on the complex 2-ball B^2
// Selects the optimal expert with strictly convex boundary stability
inline PicardRoutingProfile route_picard_modular_expert(
    float z1_real, float z1_imag,
    float z2_real, float z2_imag,
    const float* expert_centers, // num_experts * 4
    size_t num_experts
) {
    PicardRoutingProfile prof;
    prof.selected_expert_id = 0;
    prof.bergman_distance = std::numeric_limits<float>::max();

    float r_sq = z1_real * z1_real + z1_imag * z1_imag + z2_real * z2_real + z2_imag * z2_imag;
    prof.is_inside_ball = (r_sq < 1.0f);

    float denom = std::max(1e-4f, 1.0f - std::min(0.999f, r_sq));

    for (size_t e = 0; e < num_experts; ++e) {
        const float* c = expert_centers + e * 4;
        float diff_sq = (z1_real - c[0]) * (z1_real - c[0]) +
                        (z1_imag - c[1]) * (z1_imag - c[1]) +
                        (z2_real - c[2]) * (z2_real - c[2]) +
                        (z2_imag - c[3]) * (z2_imag - c[3]);

        // Bergman weighted distance
        float b_dist = diff_sq / denom;
        if (b_dist < prof.bergman_distance) {
            prof.bergman_distance = b_dist;
            prof.selected_expert_id = e;
        }
    }

    prof.routing_confidence = 1.0f / (1.0f + prof.bergman_distance);
    return prof;
}

} // namespace weft

#endif // WEFT_UNITARY_SHIMURA_H
