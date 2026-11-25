#include "csrc/cli/weft_derived_crystalline_cohomology.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    constexpr size_t dim = 128;
    std::vector<float> full_act(dim);
    std::vector<int8_t> ternary_weights(dim);

    constexpr float target_scale = 0.75f;
    for (size_t i = 0; i < dim; ++i) {
        int8_t w = static_cast<int8_t>((i % 3) - 1); // -1, 0, +1
        ternary_weights[i] = w;
        full_act[i] = target_scale * static_cast<float>(w) + 0.02f; // Slight noise
    }

    float initial_scale = 0.50f; // Underestimated scale
    float new_scale = weft::crystalline_recalibrate_scale_avx512(
        full_act.data(), ternary_weights.data(), initial_scale, dim, 2);

    // Should move closer to target_scale (0.75f)
    assert(new_scale > initial_scale);
    assert(std::abs(new_scale - target_scale) < std::abs(initial_scale - target_scale));

    std::cout << "[PASS] Derived Crystalline Recalibration Unit Test passed (Initial: " 
              << initial_scale << " -> Recalibrated: " << new_scale << " towards target " << target_scale << ").\n";
    return 0;
}
