// test_motivic_milnor_rope.cpp
// Milestone Paper 248: Verification of AVX-512 Motivic Milnor Rotary Positional Embeddings
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_motivic_milnor_rope.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_milnor_rope_rotation_kernel() {
    const size_t N = 32;
    std::vector<float> q0(N, 1.0f);
    std::vector<float> q1(N, 0.0f);
    std::vector<float> out0(N, 0.0f);
    std::vector<float> out1(N, 0.0f);

    // Rotate by 90 degrees: cos = 0, sin = 1
    float cos_t = 0.0f;
    float sin_t = 1.0f;

    weft::avx512_milnor_rope_rotate_2d(q0.data(), q1.data(), out0.data(), out1.data(), N, cos_t, sin_t);

    // Expected: out0 = 1 * 0 - 0 * 1 = 0. out1 = 1 * 1 + 0 * 0 = 1
    for (size_t i = 0; i < N; ++i) {
        assert(std::fabs(out0[i] - 0.0f) < 1e-5f);
        assert(std::fabs(out1[i] - 1.0f) < 1e-5f);
    }
}

int main() {
    std::cout << "[Paper 248] Running Motivic Milnor Rotary Positional Embeddings Unit Tests...\n";
    test_milnor_rope_rotation_kernel();
    std::cout << "[Paper 248] All unit tests PASSED successfully.\n";
    return 0;
}
