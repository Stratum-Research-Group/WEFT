// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Rigid Cohomology Recurrence Invariants (Paper 214)

#include "csrc/cli/weft_rigid_cohomology_recurrence.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_step_linear_recurrence() {
    const size_t dim = 32;
    std::vector<float> h_prev(dim, 2.0f);
    std::vector<float> x_curr(dim, 1.0f);
    std::vector<float> h_next(dim, 0.0f);

    float alpha = 0.5f;
    weft::rigid_cohomology::step_linear_recurrence_avx512(
        h_prev.data(), x_curr.data(), h_next.data(), dim, alpha
    );

    // 2.0 * 0.5 + 1.0 = 2.0
    for (size_t i = 0; i < dim; ++i) {
        assert(std::abs(h_next[i] - 2.0f) < 1e-6f);
    }
    std::cout << "[PASS] test_step_linear_recurrence" << std::endl;
}

void test_prefix_recurrence_scan() {
    const size_t dim = 16;
    const size_t tokens = 4;
    std::vector<float> x_tokens(tokens * dim, 1.0f);
    std::vector<float> h_states(tokens * dim, 0.0f);

    weft::rigid_cohomology::prefix_recurrence_scan(
        x_tokens.data(), h_states.data(), tokens, dim, 0.5f
    );

    // t=0: 1.0
    // t=1: 1.0*0.5 + 1.0 = 1.5
    // t=2: 1.5*0.5 + 1.0 = 1.75
    // t=3: 1.75*0.5 + 1.0 = 1.875
    assert(std::abs(h_states[0 * dim] - 1.0f) < 1e-5f);
    assert(std::abs(h_states[1 * dim] - 1.5f) < 1e-5f);
    assert(std::abs(h_states[2 * dim] - 1.75f) < 1e-5f);
    assert(std::abs(h_states[3 * dim] - 1.875f) < 1e-5f);
    std::cout << "[PASS] test_prefix_recurrence_scan (4 tokens verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 214: Rigid Cohomology Recurrence Unit Tests ===" << std::endl;
    test_step_linear_recurrence();
    test_prefix_recurrence_scan();
    std::cout << "All Rigid Cohomology Recurrence tests PASSED." << std::endl;
    return 0;
}
