#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_derived_floer_homology.h"

int main() {
    std::cout << "[Test] Initializing Derived Floer Homology Update SIMD Test..." << std::endl;

    const size_t count = 32;
    const float eta = 0.1f;
    const float gamma = 0.05f;

    std::vector<float> weights(count, 1.0f);
    std::vector<float> momentum(count, 0.2f);
    std::vector<float> floer_corrections(count, 0.1f);
    std::vector<float> out_weights(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        weights[i] = static_cast<float>(i + 1) * 0.25f;
        momentum[i] = static_cast<float>(i % 4) * 0.5f;
        floer_corrections[i] = static_cast<float>(i % 3) * 0.2f;
    }

    weft::avx512_floer_momentum_update(
        weights.data(),
        momentum.data(),
        floer_corrections.data(),
        out_weights.data(),
        count,
        eta,
        gamma
    );

    for (size_t i = 0; i < count; ++i) {
        float expected = weights[i] + eta * momentum[i] - gamma * floer_corrections[i];
        assert(std::fabs(out_weights[i] - expected) < 1e-5f);
    }

    std::cout << "[Test] Weight [0]: " << out_weights[0] << ", Weight [31]: " << out_weights[31] << std::endl;
    std::cout << "[Test] Derived Floer Homology Update SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
