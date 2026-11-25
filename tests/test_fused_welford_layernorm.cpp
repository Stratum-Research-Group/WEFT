#include "csrc/cli/weft_fused_welford_layernorm.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    constexpr size_t dim = 128;
    std::vector<float> input(dim);
    std::vector<float> gamma(dim, 1.0f);
    std::vector<float> beta(dim, 0.0f);
    std::vector<float> output(dim, 0.0f);

    for (size_t i = 0; i < dim; ++i) {
        input[i] = static_cast<float>(i);
    }

    weft::fused_welford_layernorm_avx512(
        input.data(), gamma.data(), beta.data(), output.data(), dim);

    float sum_out = 0.0f;
    float sum_sq_out = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        assert(!std::isnan(output[i]));
        sum_out += output[i];
        sum_sq_out += output[i] * output[i];
    }

    float mean_out = sum_out / static_cast<float>(dim);
    float var_out = (sum_sq_out / static_cast<float>(dim)) - (mean_out * mean_out);

    // Normalized: mean should be ~0.0, variance should be ~1.0
    assert(std::abs(mean_out) < 1e-4f);
    assert(std::abs(var_out - 1.0f) < 0.05f);

    std::cout << "[PASS] Fused Welford LayerNorm Unit Test passed (Mean: " 
              << mean_out << ", Var: " << var_out << ").\n";
    return 0;
}
