// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Arithmetic Intersection Theory on Deligne-Mumford Stacks (Paper 207)

#include "csrc/cli/weft_arithmetic_intersection.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

void test_intersection_symmetry() {
    weft::arithmetic_intersection::ArithmeticCycle c1{3, 1.5f};
    weft::arithmetic_intersection::ArithmeticCycle c2{5, 2.0f};

    float p12 = weft::arithmetic_intersection::compute_intersection_pairing(c1, c2);
    float p21 = weft::arithmetic_intersection::compute_intersection_pairing(c2, c1);

    assert(std::abs(p12 - p21) < 1e-6f);
    // 3*5 - 1.5*2.0 = 15.0 - 3.0 = 12.0
    assert(std::abs(p12 - 12.0f) < 1e-6f);
    std::cout << "[PASS] test_intersection_symmetry (pairing: " << p12 << ")" << std::endl;
}

void test_primitive_cycle_hodge_negative() {
    // Primitive cycle has degree 0
    weft::arithmetic_intersection::ArithmeticCycle prim{0, 4.0f};
    float self_p = weft::arithmetic_intersection::compute_intersection_pairing(prim, prim);
    // 0*0 - 4.0*4.0 = -16.0 <= 0
    assert(self_p <= 0.0f);
    assert(std::abs(self_p - (-16.0f)) < 1e-6f);
    std::cout << "[PASS] test_primitive_cycle_hodge_negative (self pairing: " << self_p << ")" << std::endl;
}

void test_batch_intersection_simd() {
    const size_t N = 32;
    std::vector<int32_t> q_deg(N, 2);
    std::vector<float> q_arch(N, 1.0f);
    std::vector<int32_t> k_deg(N, 4);
    std::vector<float> k_arch(N, 0.5f);
    std::vector<float> out(N, 0.0f);

    weft::arithmetic_intersection::batch_intersection_avx512(
        q_deg.data(), q_arch.data(),
        k_deg.data(), k_arch.data(),
        out.data(), N
    );

    for (size_t i = 0; i < N; ++i) {
        // 2*4 - 1.0*0.5 = 8.0 - 0.5 = 7.5
        assert(std::abs(out[i] - 7.5f) < 1e-5f);
    }
    std::cout << "[PASS] test_batch_intersection_simd (N=32)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 207: Arithmetic Intersection Unit Tests ===" << std::endl;
    test_intersection_symmetry();
    test_primitive_cycle_hodge_negative();
    test_batch_intersection_simd();
    std::cout << "All Arithmetic Intersection tests PASSED." << std::endl;
    return 0;
}
