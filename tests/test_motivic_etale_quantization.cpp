// test_motivic_etale_quantization.cpp
// Milestone Paper 251: Verification of AVX-512 Motivic Étale Sub-Bit Quantization
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_motivic_etale_quantization.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_scale_invariant_quantization() {
    const size_t N = 32;
    std::vector<float> x(N, 0.0f);
    // Values: +2, -2, and near 0
    x[0] = 2.0f;
    x[1] = -2.0f;
    x[2] = 0.1f;

    std::vector<float> q1(N, 0.0f);
    float scale1 = 0.0f;
    weft::avx512_scale_invariant_ternary_quantize(x.data(), q1.data(), N, 0.5f, &scale1);

    // Scale by factor of 10.0f
    std::vector<float> x_scaled(N, 0.0f);
    for (size_t i = 0; i < N; ++i) x_scaled[i] = x[i] * 10.0f;

    std::vector<float> q2(N, 0.0f);
    float scale2 = 0.0f;
    weft::avx512_scale_invariant_ternary_quantize(x_scaled.data(), q2.data(), N, 0.5f, &scale2);

    // Invariant: Quantized ternary signs are strictly identical
    for (size_t i = 0; i < N; ++i) {
        assert(q1[i] == q2[i]);
    }
    // Scale scales linearly by 10
    assert(std::fabs(scale2 - scale1 * 10.0f) < 1e-4f);
}

int main() {
    std::cout << "[Paper 251] Running Motivic Étale Sub-Bit Quantization Unit Tests...\n";
    test_scale_invariant_quantization();
    std::cout << "[Paper 251] All unit tests PASSED successfully.\n";
    return 0;
}
