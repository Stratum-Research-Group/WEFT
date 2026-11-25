#include "csrc/cli/weft_motivic_shimura_curves.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    constexpr size_t dim = 128;
    std::vector<int8_t> blinded_weights(dim);
    std::vector<float> input(dim, 1.0f);
    float output = 0.0f;

    for (size_t i = 0; i < dim; ++i) {
        blinded_weights[i] = static_cast<int8_t>((i % 3) - 1); // {-1, 0, +1}
    }

    constexpr float key = 2.71828f;
    constexpr float scale = 0.5f;

    weft::shimura_blinded_gemv_avx512(
        blinded_weights.data(), input.data(), &output, dim, key, scale);

    // Compute reference output
    float ref = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        ref += input[i] * static_cast<float>(blinded_weights[i]) * scale;
    }

    assert(std::abs(output - ref) < 1e-4f);

    std::cout << "[PASS] Motivic Shimura Curves Blinded GEMV Unit Test passed (Output: " 
              << output << " == Ref: " << ref << ").\n";
    return 0;
}
