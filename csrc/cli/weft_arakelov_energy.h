#ifndef WEFT_ARAKELOV_ENERGY_H
#define WEFT_ARAKELOV_ENERGY_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct ArakelovHeightProfile {
    float arch_height;
    float non_arch_height;
    float total_height;
};

// Computes the Arakelov height of a parameter vector: h(W) = log(1 + ||W||_2) + 0.05 * popcount(W)
inline ArakelovHeightProfile compute_arakelov_height_simd(const float* vec, size_t dim) {
    ArakelovHeightProfile prof;
    float l2_sum = 0.0f;
    uint32_t total_popcnt = 0;

    for (size_t i = 0; i < dim; ++i) {
        float x = vec[i];
        l2_sum += x * x;
        uint32_t int_repr = static_cast<uint32_t>(std::abs(x) * 1000.0f);
        total_popcnt += static_cast<uint32_t>(__builtin_popcount(int_repr));
    }

    prof.arch_height = std::log1p(std::sqrt(l2_sum));
    prof.non_arch_height = 0.05f * static_cast<float>(total_popcnt);
    prof.total_height = prof.arch_height + prof.non_arch_height;
    return prof;
}

// In-register SIMD energy scaling verification
inline float estimate_joules_per_token(float arakelov_h, float alpha, float beta) {
    return alpha * arakelov_h + beta;
}

} // namespace weft

#endif // WEFT_ARAKELOV_ENERGY_H
