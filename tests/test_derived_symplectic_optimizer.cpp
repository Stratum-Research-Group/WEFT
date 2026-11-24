// test_derived_symplectic_optimizer.cpp
// Milestone Paper 236: Verification of AVX-512 Derived Symplectic Leapfrog Optimizer Updates
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_derived_symplectic_optimizer.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_symplectic_step() {
    const size_t N = 32;
    std::vector<float> q(N, 1.0f);
    std::vector<float> p(N, 0.5f);
    std::vector<float> grad(N, 0.2f);

    float eps = 0.1f;
    // half_eps = 0.05
    // p_half = 0.5 - 0.05 * 0.2 = 0.5 - 0.01 = 0.49
    // q_next = 1.0 + 0.1 * 0.49 = 1.0 + 0.049 = 1.049
    weft::avx512_symplectic_leapfrog_step(q.data(), p.data(), grad.data(), N, eps);

    for (size_t i = 0; i < N; ++i) {
        assert(std::fabs(p[i] - 0.49f) < 1e-5f);
        assert(std::fabs(q[i] - 1.049f) < 1e-5f);
    }
}

int main() {
    std::cout << "[Paper 236] Running Derived Symplectic Optimizer Unit Tests...\n";
    test_symplectic_step();
    std::cout << "[Paper 236] All unit tests PASSED successfully.\n";
    return 0;
}
