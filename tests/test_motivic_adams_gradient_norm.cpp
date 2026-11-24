// test_motivic_adams_gradient_norm.cpp
// Milestone Paper 245: Verification of AVX-512 Motivic Adams Gradient Normalization
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_motivic_adams_gradient_norm.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_adams_gradient_norm_kernel() {
    const size_t N = 32;
    std::vector<float> grad(N, 5.0f);
    std::vector<float> out(N, 0.0f);

    weft::AdamsNormConfig cfg{2.0f, 0.1f};
    weft::avx512_motivic_adams_normalize_grad(grad.data(), out.data(), N, cfg);

    // num = 2.0 * 5.0 = 10.0
    // denom = 1.0 + 0.1 * 2.0 * 5.0 = 1.0 + 1.0 = 2.0
    // expected = 10.0 / 2.0 = 5.0
    for (size_t i = 0; i < N; ++i) {
        assert(std::fabs(out[i] - 5.0f) < 1e-4f);
    }

    // Outlier gradient test
    grad[0] = 1000.0f;
    weft::avx512_motivic_adams_normalize_grad(grad.data(), out.data(), N, cfg);
    // As g -> infty, out -> 1.0 / lambda = 1.0 / 0.1 = 10.0
    // For 1000: num = 2000, denom = 1 + 200 = 201. out = 2000 / 201 = 9.9502
    assert(std::fabs(out[0] - (2000.0f / 201.0f)) < 1e-4f);
}

int main() {
    std::cout << "[Paper 245] Running Motivic Adams Gradient Normalization Unit Tests...\n";
    test_adams_gradient_norm_kernel();
    std::cout << "[Paper 245] All unit tests PASSED successfully.\n";
    return 0;
}
