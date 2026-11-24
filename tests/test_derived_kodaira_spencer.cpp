// test_derived_kodaira_spencer.cpp
// Milestone Paper 246: Verification of AVX-512 Derived Kodaira-Spencer Layer Pruning
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_derived_kodaira_spencer.h"
#include <iostream>
#include <cassert>
#include <vector>

void test_kodaira_spencer_layer_pruning() {
    const size_t N = 32;
    std::vector<float> in_x(N, 2.0f);
    std::vector<float> delta_x(N, 0.01f); // Tiny deformation

    float tau_ks = 0.01f;
    // norm_sq = 32 * 4 = 128. delta_sq = 32 * 0.0001 = 0.0032. energy = 0.0032 / 128 = 0.000025 < 0.01 -> skip
    bool skip = weft::avx512_should_skip_layer_ks(in_x.data(), delta_x.data(), N, tau_ks);
    assert(skip);

    // Large deformation
    std::fill(delta_x.begin(), delta_x.end(), 1.0f);
    // delta_sq = 32 * 1 = 32. energy = 32 / 128 = 0.25 > 0.01 -> do not skip
    bool no_skip = weft::avx512_should_skip_layer_ks(in_x.data(), delta_x.data(), N, tau_ks);
    assert(!no_skip);
}

int main() {
    std::cout << "[Paper 246] Running Derived Kodaira-Spencer Dynamic Layer Pruning Unit Tests...\n";
    test_kodaira_spencer_layer_pruning();
    std::cout << "[Paper 246] All unit tests PASSED successfully.\n";
    return 0;
}
