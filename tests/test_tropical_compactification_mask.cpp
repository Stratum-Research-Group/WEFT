// test_tropical_compactification_mask.cpp
// Milestone Paper 238: Verification of AVX-512 Tropical Compactification for Sparse Attention
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_tropical_compactification_mask.h"
#include <iostream>
#include <cassert>
#include <vector>

void test_tropical_mask_kernel() {
    const size_t N = 32;
    std::vector<float> scores(N, 0.0f);
    // Even indices >= 5.0, odd indices < 5.0
    for (size_t i = 0; i < N; ++i) {
        scores[i] = (i % 2 == 0) ? 10.0f : 2.0f;
    }

    std::vector<uint16_t> masks(2, 0);
    weft::avx512_tropical_sparse_mask(scores.data(), masks.data(), N, 5.0f);

    // Expected bitmask for alternating pattern (0b0101010101010101 = 0x5555)
    assert(masks[0] == 0x5555);
    assert(masks[1] == 0x5555);
}

int main() {
    std::cout << "[Paper 238] Running Tropical Compactification Sparse Mask Unit Tests...\n";
    test_tropical_mask_kernel();
    std::cout << "[Paper 238] All unit tests PASSED successfully.\n";
    return 0;
}
