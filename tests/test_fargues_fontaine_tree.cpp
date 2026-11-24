// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Fargues-Fontaine Speculative Memory Bundles (Paper 221)

#include "csrc/cli/weft_fargues_fontaine_tree.h"
#include <iostream>
#include <cassert>
#include <vector>

void test_filter_speculative_branches() {
    const size_t N = 32;
    std::vector<float> degrees(N);
    std::vector<float> ranks(N, 2.0f);
    std::vector<uint32_t> retained(N, 0);

    for (size_t i = 0; i < N; ++i) {
        degrees[i] = static_cast<float>(i);
    }

    // slope = i / 2.0. Threshold = 5.0 -> i >= 10
    float min_slope = 5.0f;
    size_t count = weft::fargues_fontaine_tree::filter_speculative_branches_avx512(
        degrees.data(), ranks.data(), N, min_slope, retained.data()
    );

    assert(count == (N - 10)); // 32 - 10 = 22 branches retained
    for (size_t k = 0; k < count; ++k) {
        uint32_t idx = retained[k];
        assert(degrees[idx] / ranks[idx] >= min_slope);
    }
    std::cout << "[PASS] test_filter_speculative_branches (retained " << count << " branches)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 221: Fargues-Fontaine Speculative Bundles Unit Tests ===" << std::endl;
    test_filter_speculative_branches();
    std::cout << "All Fargues-Fontaine Speculative Bundles tests PASSED." << std::endl;
    return 0;
}
