// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Liquid Softmax Normalization Invariants (Paper 219)

#include "csrc/cli/weft_liquid_softmax.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_liquid_softmax_basic() {
    const size_t N = 32;
    std::vector<float> logits(N);
    std::vector<float> probs(N, 0.0f);

    for (size_t i = 0; i < N; ++i) {
        logits[i] = static_cast<float>(i);
    }

    weft::liquid_softmax::compute_liquid_softmax_avx512(
        logits.data(), probs.data(), N, 1.0f
    );

    float sum = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        assert(probs[i] >= 0.0f);
        assert(probs[i] <= 1.0f);
        sum += probs[i];
    }
    assert(std::abs(sum - 1.0f) < 1e-5f);
    std::cout << "[PASS] test_liquid_softmax_basic (sum == 1.0 verified)" << std::endl;
}

void test_liquid_softmax_extreme_logits() {
    // Tests stability under extreme logits (1000.0f) which would overflow standard exp
    std::vector<float> logits = {1000.0f, 1000.0f, 999.0f};
    std::vector<float> probs(3, 0.0f);

    weft::liquid_softmax::compute_liquid_softmax_avx512(
        logits.data(), probs.data(), 3, 1.0f
    );

    assert(!std::isnan(probs[0]));
    assert(!std::isinf(probs[0]));
    assert(probs[0] > 0.0f);
    std::cout << "[PASS] test_liquid_softmax_extreme_logits (overflow prevented)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 219: Liquid Softmax Unit Tests ===" << std::endl;
    test_liquid_softmax_basic();
    test_liquid_softmax_extreme_logits();
    std::cout << "All Liquid Softmax tests PASSED." << std::endl;
    return 0;
}
