#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_derived_critical_loci.h"

int main() {
    std::cout << "[Test] Initializing Derived Critical Loci Smoothing SIMD Test..." << std::endl;

    const size_t count = 32;
    const float gamma = 0.2f;

    std::vector<float> raw_gradients(count, 5.0f);
    std::vector<float> vanishing_cycles(count, 1.0f);
    std::vector<float> out_gradients(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        raw_gradients[i] = static_cast<float>(i + 1) * 0.5f;
        vanishing_cycles[i] = static_cast<float>(i % 4) * 0.1f;
    }

    weft::avx512_vanishing_cycle_smoothing(
        raw_gradients.data(),
        vanishing_cycles.data(),
        out_gradients.data(),
        count,
        gamma
    );

    for (size_t i = 0; i < count; ++i) {
        float expected = raw_gradients[i] - gamma * vanishing_cycles[i];
        assert(std::fabs(out_gradients[i] - expected) < 1e-5f);
    }

    std::cout << "[Test] Out Gradient [0]: " << out_gradients[0] << std::endl;
    std::cout << "[Test] Out Gradient [31]: " << out_gradients[31] << std::endl;
    std::cout << "[Test] Derived Critical Loci Smoothing SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
