#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_derived_gauss_manin.h"

int main() {
    std::cout << "[Test] Initializing Derived Gauss-Manin Parallel Transport SIMD Test..." << std::endl;

    const size_t count = 32;
    const float ds = 0.05f;

    std::vector<float> weights(count, 1.0f);
    std::vector<float> connection(count, 0.2f);
    std::vector<float> out_weights(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        weights[i] = static_cast<float>(i + 1) * 0.5f;
        connection[i] = static_cast<float>(i % 5) * 0.1f;
    }

    weft::avx512_gauss_manin_transport(
        weights.data(),
        connection.data(),
        out_weights.data(),
        count,
        ds
    );

    for (size_t i = 0; i < count; ++i) {
        float expected = weights[i] - ds * connection[i] * weights[i];
        assert(std::fabs(out_weights[i] - expected) < 1e-5f);
    }

    std::cout << "[Test] Transported Weight [0]: " << out_weights[0] << std::endl;
    std::cout << "[Test] Transported Weight [31]: " << out_weights[31] << std::endl;
    std::cout << "[Test] Derived Gauss-Manin Parallel Transport SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
