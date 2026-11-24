// test_clifford_attention.cpp
// Numerical validation of Cl(3,0) Clifford multivector multiplication and rotor norm invariance.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_clifford_attention.h"
#include <iostream>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_clifford_attention..." << std::endl;

    // Construct a pure vector x = 1.0 * e1 + 2.0 * e2 + 3.0 * e3
    weft::Multivector8 x;
    x.v[1] = 1.0f;
    x.v[2] = 2.0f;
    x.v[3] = 3.0f;

    float orig_norm_sq = x.norm_sq();
    std::cout << "[TEST] Original Vector Norm Squared: " << orig_norm_sq << std::endl;

    // Construct a planar rotor R = cos(theta/2) + sin(theta/2) * e12 (theta = pi/4)
    float theta = 3.14159265f / 4.0f;
    weft::Multivector8 R;
    R.v[0] = std::cos(theta / 2.0f);
    R.v[4] = std::sin(theta / 2.0f);

    float r_norm_sq = R.norm_sq();
    std::cout << "[TEST] Rotor Norm Squared: " << r_norm_sq << std::endl;
    assert(std::abs(r_norm_sq - 1.0f) < 1e-5f);

    // Apply rotor rotation x' = R * x * R^dagger
    weft::Multivector8 x_rot = weft::clifford_rotate(R, x);

    float rot_norm_sq = x_rot.norm_sq();
    std::cout << "[TEST] Rotated Vector Norm Squared: " << rot_norm_sq << std::endl;

    float rel_diff = std::abs(rot_norm_sq - orig_norm_sq) / orig_norm_sq;
    std::cout << "[TEST] Relative Norm Difference: " << rel_diff << std::endl;
    assert(rel_diff < 1e-5f);

    // Verify z-component (e3) is unchanged under e12 rotation
    assert(std::abs(x_rot.v[3] - 3.0f) < 1e-5f);

    std::cout << "[TEST] All Clifford Spinor Attention tests passed successfully!" << std::endl;
    return 0;
}
