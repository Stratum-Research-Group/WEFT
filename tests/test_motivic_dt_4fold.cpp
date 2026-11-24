// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Motivic Donaldson-Thomas Calabi-Yau 4-Fold Invariants (Paper 226)

#include "csrc/cli/weft_motivic_dt_4fold.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_dot_product_4bit() {
    const size_t N = 32;
    std::vector<uint8_t> packed(N / 2); // 16 bytes for 32 nibbles
    std::vector<float> acts(N, 1.0f);

    // Set nibbles to 9 (w = 9 - 8 = +1)
    for (size_t i = 0; i < N / 2; ++i) {
        packed[i] = 0x99; // both nibbles = 9
    }

    float res = weft::motivic_dt_4fold::dot_product_4bit_avx512(
        packed.data(), acts.data(), N
    );

    // 32 * (+1.0) * 1.0 = 32.0
    assert(std::abs(res - 32.0f) < 1e-5f);
    std::cout << "[PASS] test_dot_product_4bit (N=32, result 32.0 verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 226: Motivic DT 4-Fold Unit Tests ===" << std::endl;
    test_dot_product_4bit();
    std::cout << "All Motivic DT 4-Fold tests PASSED." << std::endl;
    return 0;
}
