// test_octonion_attention.cpp
// Validation of Octonionic multiplication, norm multiplicativity, and alternativity.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_octonion_attention.h"
#include <iostream>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_octonion_attention..." << std::endl;

    weft::Octonion x;
    weft::Octonion y;

    // x = 1 + 2*e1 + 3*e2
    x.x[0] = 1.0f; x.x[1] = 2.0f; x.x[2] = 3.0f;

    // y = 2 + e4 + e5
    y.x[0] = 2.0f; y.x[4] = 1.0f; y.x[5] = 1.0f;

    float norm_x_sq = x.norm_sq();
    float norm_y_sq = y.norm_sq();

    weft::Octonion xy = weft::octonion_mul(x, y);
    float norm_xy_sq = xy.norm_sq();

    std::cout << "[TEST] ||x||^2 = " << norm_x_sq << ", ||y||^2 = " << norm_y_sq << std::endl;
    std::cout << "[TEST] ||xy||^2 = " << norm_xy_sq << " (Expected " << norm_x_sq * norm_y_sq << ")" << std::endl;

    float rel_diff = std::abs(norm_xy_sq - (norm_x_sq * norm_y_sq)) / (norm_x_sq * norm_y_sq);
    assert(rel_diff < 1e-5f);
    std::cout << "[TEST] Norm Multiplicativity Confirmed." << std::endl;

    // Test Alternativity: [x, x, y] == 0
    weft::Octonion assoc_xxy = weft::octonion_associator(x, x, y);
    float assoc_norm = std::sqrt(assoc_xxy.norm_sq());
    std::cout << "[TEST] Alternativity Associator Norm ||[x, x, y]|| = " << assoc_norm << " (Expected 0.0)" << std::endl;
    assert(assoc_norm < 1e-5f);

    std::cout << "[TEST] All Octonionic Non-Associative Attention tests passed successfully!" << std::endl;
    return 0;
}
