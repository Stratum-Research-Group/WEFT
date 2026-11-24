// test_wasserstein_speculation.cpp
// Validation of 1D Wasserstein-1 distance metric properties and speculative acceptance.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_wasserstein_speculation.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_wasserstein_speculation..." << std::endl;

    const int K = 8;
    std::vector<float> p(K, 0.0f);
    std::vector<float> q(K, 0.0f);

    // Distribution P: mass concentrated at index 0
    p[0] = 1.0f;
    // Distribution Q: mass concentrated at index 1
    q[1] = 1.0f;

    // Expected W1 distance between delta_0 and delta_1 is exactly 1.0 (mass moved by 1 step)
    float w1_01 = weft::wasserstein_distance_1d_avx512(p.data(), q.data(), K);
    std::cout << "[TEST] W1(delta_0, delta_1) = " << w1_01 << " (Expected 1.0)" << std::endl;
    assert(std::abs(w1_01 - 1.0f) < 1e-6f);

    // Self distance W1(P, P) = 0.0
    float w1_self = weft::wasserstein_distance_1d_avx512(p.data(), p.data(), K);
    std::cout << "[TEST] W1(delta_0, delta_0) = " << w1_self << " (Expected 0.0)" << std::endl;
    assert(std::abs(w1_self) < 1e-6f);

    // Symmetry check: W1(P, Q) == W1(Q, P)
    float w1_10 = weft::wasserstein_distance_1d_avx512(q.data(), p.data(), K);
    assert(std::abs(w1_01 - w1_10) < 1e-6f);

    // Speculative tolerance verification
    bool accepted = weft::verify_speculative_wasserstein(p.data(), q.data(), K, 1.5f);
    assert(accepted == true);

    bool rejected = weft::verify_speculative_wasserstein(p.data(), q.data(), K, 0.5f);
    assert(rejected == false);

    std::cout << "[TEST] All Optimal Transport Wasserstein Speculation tests passed successfully!" << std::endl;
    return 0;
}
