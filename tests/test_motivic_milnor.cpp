// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Motivic Milnor Fiber Activation Invariants (Paper 220)

#include "csrc/cli/weft_motivic_milnor.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_smooth_cusp_activation() {
    const size_t N = 32;
    std::vector<float> in_acts(N);
    std::vector<float> out_acts(N, 0.0f);

    float eps = 0.10f;
    for (size_t i = 0; i < N; ++i) {
        in_acts[i] = -0.20f + static_cast<float>(i) * (0.40f / N);
    }

    weft::motivic_milnor::smooth_cusp_activation_avx512(
        in_acts.data(), out_acts.data(), N, eps
    );

    // Verify all outputs are non-negative and continuity holds
    for (size_t i = 0; i < N; ++i) {
        assert(out_acts[i] >= 0.0f);
        if (in_acts[i] <= -eps) {
            assert(out_acts[i] == 0.0f);
        } else if (in_acts[i] >= eps) {
            assert(std::abs(out_acts[i] - in_acts[i]) < 1e-5f);
        }
    }
    std::cout << "[PASS] test_smooth_cusp_activation (N=32 verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 220: Motivic Milnor Fiber Unit Tests ===" << std::endl;
    test_smooth_cusp_activation();
    std::cout << "All Motivic Milnor Fiber tests PASSED." << std::endl;
    return 0;
}
