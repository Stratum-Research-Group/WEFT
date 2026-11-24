// test_tropical_amoeba_softmax.cpp
// Milestone Paper 247: Verification of AVX-512 Tropical Amoeba Softmax Normalization
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_tropical_amoeba_softmax.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_tropical_amoeba_softmax() {
    const size_t N = 32;
    std::vector<float> logits(N, 0.0f);
    // Two top logits at 10.0f, rest at 0.0f
    logits[0] = 10.0f;
    logits[1] = 10.0f;

    std::vector<float> probs(N, 0.0f);
    float alpha = 0.5f;

    weft::avx512_tropical_amoeba_softmax(logits.data(), probs.data(), N, alpha);

    // For top logits (diff = 0): phi = 1.0. For others (diff = -10): phi = max(0, 1 - 5) = 0.
    // Total sum = 1.0 + 1.0 = 2.0. Each top prob = 0.5f.
    assert(std::fabs(probs[0] - 0.5f) < 1e-5f);
    assert(std::fabs(probs[1] - 0.5f) < 1e-5f);
    for (size_t i = 2; i < N; ++i) {
        assert(std::fabs(probs[i] - 0.0f) < 1e-5f);
    }
}

int main() {
    std::cout << "[Paper 247] Running Tropical Amoeba Softmax Normalization Unit Tests...\n";
    test_tropical_amoeba_softmax();
    std::cout << "[Paper 247] All unit tests PASSED successfully.\n";
    return 0;
}
