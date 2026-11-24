// test_lie_group_attention.cpp
// Validation of so(3) Lie bracket antisymmetry and SO(3) matrix exponential determinant.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_lie_group_attention.h"
#include <iostream>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_lie_group_attention..." << std::endl;

    weft::LieAlgebraSO3 X(1.0f, 0.0f, 0.0f); // Rotation about x-axis
    weft::LieAlgebraSO3 Y(0.0f, 1.0f, 0.0f); // Rotation about y-axis

    // Test Lie bracket antisymmetry: [X, Y] = -[Y, X]
    weft::LieAlgebraSO3 comm_XY = weft::lie_bracket_so3(X, Y);
    weft::LieAlgebraSO3 comm_YX = weft::lie_bracket_so3(Y, X);

    std::cout << "[TEST] [X, Y] = (" << comm_XY.w[0] << ", " << comm_XY.w[1] << ", " << comm_XY.w[2] << ")" << std::endl;
    assert(std::abs(comm_XY.w[2] - 1.0f) < 1e-6f); // x x y = z

    for (int i = 0; i < 3; ++i) {
        assert(std::abs(comm_XY.w[i] + comm_YX.w[i]) < 1e-6f);
    }
    std::cout << "[TEST] Lie Bracket Antisymmetry Verified." << std::endl;

    // Test exponential map exp(X) in SO(3): det(exp(X)) == 1.0
    weft::LieAlgebraSO3 Z(0.3f, -0.4f, 0.5f);
    weft::LieGroupSO3 R = weft::exponential_map_so3(Z);
    float det = R.determinant();
    std::cout << "[TEST] det(exp(Z)) = " << det << " (Expected 1.0)" << std::endl;
    assert(std::abs(det - 1.0f) < 1e-5f);

    std::cout << "[TEST] All Lie Group Gauge Attention tests passed successfully!" << std::endl;
    return 0;
}
