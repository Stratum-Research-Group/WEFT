#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_motivic_slice_quantization.h"

int main() {
    std::cout << "[Test] Initializing Motivic Slice Precision Allocation SIMD Test..." << std::endl;

    const size_t num_layers = 32;
    const float b_min = 0.5f;
    const float alpha = 0.25f;

    std::vector<float> sensitivities(num_layers, 0.0f);
    std::vector<float> out_precision(num_layers, 0.0f);

    for (size_t i = 0; i < num_layers; ++i) {
        sensitivities[i] = static_cast<float>(i);
    }

    weft::avx512_motivic_slice_allocation(
        sensitivities.data(),
        out_precision.data(),
        num_layers,
        b_min,
        alpha
    );

    for (size_t i = 0; i < num_layers; ++i) {
        float expected = b_min + alpha * sensitivities[i];
        assert(std::fabs(out_precision[i] - expected) < 1e-5f);
        assert(out_precision[i] >= b_min);
    }

    std::cout << "[Test] Precision Layer [0]: " << out_precision[0] << std::endl;
    std::cout << "[Test] Precision Layer [31]: " << out_precision[31] << std::endl;
    std::cout << "[Test] Motivic Slice Precision Allocation SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
