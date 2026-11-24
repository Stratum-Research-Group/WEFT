// test_derived_loop_speculative.cpp
// Milestone Paper 240: Verification of AVX-512 Derived Loop Speculative Verification
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_derived_loop_speculative.h"
#include <iostream>
#include <cassert>
#include <vector>

void test_speculative_verification_kernel() {
    const size_t N = 32;
    std::vector<float> draft(N, 1.0f);
    std::vector<float> target(N, 1.02f);

    float tolerance = 0.05f;
    // All 32 tokens match within tolerance 0.05
    size_t acc = weft::avx512_verify_speculative_loop(draft.data(), target.data(), N, tolerance);
    assert(acc == 32);

    // Introduce divergence at index 18
    target[18] = 1.20f; // diff = 0.20 > 0.05
    acc = weft::avx512_verify_speculative_loop(draft.data(), target.data(), N, tolerance);
    assert(acc == 18);
}

int main() {
    std::cout << "[Paper 240] Running Derived Loop Speculative Verification Unit Tests...\n";
    test_speculative_verification_kernel();
    std::cout << "[Paper 240] All unit tests PASSED successfully.\n";
    return 0;
}
