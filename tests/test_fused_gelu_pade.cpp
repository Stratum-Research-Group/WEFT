#include "csrc/cli/weft_fused_gelu_pade.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

float reference_gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

int main() {
    constexpr size_t size = 128;
    std::vector<float> input(size);
    std::vector<float> output(size, 0.0f);

    for (size_t i = 0; i < size; ++i) {
        input[i] = -4.0f + 8.0f * (static_cast<float>(i) / static_cast<float>(size - 1));
    }

    weft::fused_gelu_pade_avx512(input.data(), output.data(), size);

    float max_err = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float ref = reference_gelu(input[i]);
        float err = std::abs(output[i] - ref);
        if (err > max_err) max_err = err;
        assert(!std::isnan(output[i]));
    }

    // Minimax rational approximation matches within 0.02 across the entire range
    assert(max_err < 0.025f);

    std::cout << "[PASS] Fused GeLU Padé Unit Test passed (Max error vs exact: " 
              << max_err << ").\n";
    return 0;
}
