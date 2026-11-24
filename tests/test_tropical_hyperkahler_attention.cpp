// test_tropical_hyperkahler_attention.cpp
// Milestone Paper 244: Verification of AVX-512 Tropical Hyperkähler Attention Routing
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_tropical_hyperkahler_attention.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_hyperkahler_rotation_kernel() {
    const size_t num_quads = 8;
    std::vector<float> in_q(num_quads * 4, 0.0f);
    // Set each quad to (1, 0, 0, 0)
    for (size_t i = 0; i < num_quads; ++i) {
        in_q[i * 4 + 0] = 1.0f;
    }

    std::vector<float> out_q(num_quads * 4, 0.0f);

    // Rotate by pure I unit: (0, 1, 0, 0)
    weft::QuaternionRotation rot{0.0f, 1.0f, 0.0f, 0.0f};
    weft::avx512_hyperkahler_rotate_quads(in_q.data(), out_q.data(), num_quads, rot);

    // Expected result: q0 -> 0, q1 -> 1, q2 -> 0, q3 -> 0
    for (size_t i = 0; i < num_quads; ++i) {
        assert(std::fabs(out_q[i * 4 + 0] - 0.0f) < 1e-5f);
        assert(std::fabs(out_q[i * 4 + 1] - 1.0f) < 1e-5f);
        assert(std::fabs(out_q[i * 4 + 2] - 0.0f) < 1e-5f);
        assert(std::fabs(out_q[i * 4 + 3] - 0.0f) < 1e-5f);
    }
}

int main() {
    std::cout << "[Paper 244] Running Tropical Hyperkähler Attention Routing Unit Tests...\n";
    test_hyperkahler_rotation_kernel();
    std::cout << "[Paper 244] All unit tests PASSED successfully.\n";
    return 0;
}
