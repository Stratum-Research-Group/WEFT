// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Voevodsky Slice Filtration Invariants (Paper 212)

#include "csrc/cli/weft_voevodsky_slice.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_attenuate_residual_stream() {
    const size_t N = 48;
    std::vector<float> in_stream(N, 10.0f);
    std::vector<uint32_t> strata(N);
    std::vector<float> out_stream(N, 0.0f);

    for (size_t i = 0; i < N; ++i) {
        strata[i] = static_cast<uint32_t>(i % 3); // 0, 1, 2
    }

    weft::voevodsky_slice::attenuate_residual_stream_avx512(
        in_stream.data(), strata.data(), out_stream.data(), N, 0.25f
    );

    for (size_t i = 0; i < N; ++i) {
        if (strata[i] == 0) {
            assert(std::abs(out_stream[i] - 10.0f) < 1e-5f);
        } else if (strata[i] == 1) {
            assert(std::abs(out_stream[i] - 2.5f) < 1e-5f);
        } else {
            assert(std::abs(out_stream[i] - 0.0f) < 1e-5f);
        }
    }
    std::cout << "[PASS] test_attenuate_residual_stream (N=48, 3 strata verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 212: Voevodsky Slice Filtration Unit Tests ===" << std::endl;
    test_attenuate_residual_stream();
    std::cout << "All Voevodsky Slice Filtration tests PASSED." << std::endl;
    return 0;
}
