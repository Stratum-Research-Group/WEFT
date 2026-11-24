// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Artin Approximation Quantization Invariants (Paper 222)

#include "csrc/cli/weft_artin_approximation.h"
#include <iostream>
#include <cassert>
#include <vector>

void test_slice_weights_artin() {
    const size_t N = 32;
    std::vector<float> in(N);
    std::vector<float> out(N, 0.0f);

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) in[i] = 0.8f;
        else if (i % 3 == 1) in[i] = -0.8f;
        else in[i] = 0.1f;
    }

    weft::artin_approximation::slice_weights_artin_avx512(in.data(), out.data(), N, 1.0f);

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) assert(out[i] == 1.0f);
        else if (i % 3 == 1) assert(out[i] == -1.0f);
        else assert(out[i] == 0.0f);
    }
    std::cout << "[PASS] test_slice_weights_artin (ternary root quantization verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 222: Artin Approximation Unit Tests ===" << std::endl;
    test_slice_weights_artin();
    std::cout << "All Artin Approximation tests PASSED." << std::endl;
    return 0;
}
