#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_derived_taq_activation.h"

int main() {
    std::cout << "[Test] Initializing Derived TAQ Activation SIMD Test..." << std::endl;

    const size_t count = 32;
    const float slope = 0.8f;
    const float bias = 0.2f;
    const float min_val = 0.0f;

    std::vector<float> in_vec(count, 0.0f);
    std::vector<float> out_vec(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        in_vec[i] = static_cast<float>(i) * 0.25f - 2.0f;
    }

    weft::avx512_taq_linear_activation(
        in_vec.data(),
        out_vec.data(),
        count,
        slope,
        bias,
        min_val
    );

    for (size_t i = 0; i < count; ++i) {
        float expected = slope * in_vec[i] + bias;
        if (expected < min_val) expected = min_val;
        assert(std::fabs(out_vec[i] - expected) < 1e-5f);
        assert(out_vec[i] >= min_val);
    }

    std::cout << "[Test] Out [0]: " << out_vec[0] << " (clamped to 0)" << std::endl;
    std::cout << "[Test] Out [31]: " << out_vec[31] << std::endl;
    std::cout << "[Test] Derived TAQ Activation SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
