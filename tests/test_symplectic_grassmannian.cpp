// test_symplectic_grassmannian.cpp
// Validation of Symplectic Stiefel frame canonical 2-form invariants.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_symplectic_grassmannian.h"
#include <iostream>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_symplectic_grassmannian..." << std::endl;

    const int d = 32;
    const int k = 2; // 2k = 4 columns

    weft::SymplecticFrame frame(d, k);

    // Set canonical symplectic basis:
    // col 0: q_0 = e_0, p_0 = 0
    // col 1: q_1 = e_1, p_1 = 0
    // col 2: q_2 = 0,   p_2 = e_0
    // col 3: q_3 = 0,   p_3 = e_1
    frame.Q[0 * d + 0] = 1.0f; // q0
    frame.Q[1 * d + 1] = 1.0f; // q1
    frame.P[2 * d + 0] = 1.0f; // p0
    frame.P[3 * d + 1] = 1.0f; // p1

    // Check canonical symplectic relations:
    // omega(col 0, col 2) = q0^T p0 - p0^T q0 = 1.0 - 0.0 = 1.0
    float omega_02 = weft::symplectic_inner_product_avx512(frame, 0, 2);
    std::cout << "[TEST] omega(col 0, col 2) = " << omega_02 << " (Expected 1.0)" << std::endl;
    assert(std::abs(omega_02 - 1.0f) < 1e-6f);

    // Skew-symmetry: omega(col 2, col 0) = -1.0
    float omega_20 = weft::symplectic_inner_product_avx512(frame, 2, 0);
    std::cout << "[TEST] omega(col 2, col 0) = " << omega_20 << " (Expected -1.0)" << std::endl;
    assert(std::abs(omega_20 + 1.0f) < 1e-6f);

    // Orthogonal columns: omega(col 0, col 1) = 0.0
    float omega_01 = weft::symplectic_inner_product_avx512(frame, 0, 1);
    std::cout << "[TEST] omega(col 0, col 1) = " << omega_01 << " (Expected 0.0)" << std::endl;
    assert(std::abs(omega_01) < 1e-6f);

    std::cout << "[TEST] All Symplectic Grassmannian tests passed successfully!" << std::endl;
    return 0;
}
