#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_motivic_lfunctions_precision.h"

int main() {
    std::cout << "[Test] Initializing Motivic L-Functions Precision Allocation SIMD Test..." << std::endl;

    const size_t count = 32;
    const float base_bits = 2.0f;
    const float min_bits = 0.5f;
    const float mean_regulator = 10.0f;
    const float inv_beta = 0.2f;

    std::vector<float> regulators(count, 0.0f);
    std::vector<float> out_precision(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        regulators[i] = static_cast<float>(i);
    }

    weft::avx512_motivic_precision_allocation(
        regulators.data(),
        out_precision.data(),
        count,
        base_bits,
        min_bits,
        mean_regulator,
        inv_beta
    );

    for (size_t i = 0; i < count; ++i) {
        float expected = base_bits + inv_beta * (regulators[i] - mean_regulator);
        if (expected < min_bits) expected = min_bits;
        assert(std::fabs(out_precision[i] - expected) < 1e-5f);
        assert(out_precision[i] >= min_bits);
    }

    std::cout << "[Test] Precision Sample [0]: " << out_precision[0] << " (clamped to min)" << std::endl;
    std::cout << "[Test] Precision Sample [31]: " << out_precision[31] << std::endl;
    std::cout << "[Test] Motivic L-Functions Precision Allocation SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
