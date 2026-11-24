// test_subriemannian_attention.cpp
// Validation of Heisenberg Carnot group operations and Koranyi sub-Riemannian gauge distance.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_subriemannian_attention.h"
#include <iostream>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_subriemannian_attention..." << std::endl;

    weft::HeisenbergPoint u(1.0f, 2.0f, 3.0f);
    weft::HeisenbergPoint v(2.0f, 4.0f, 5.0f);

    // Self distance d_CC(u, u) = 0
    float d_self = weft::subriemannian_distance(u, u);
    std::cout << "[TEST] Sub-Riemannian Self Distance d_CC(u, u) = " << d_self << " (Expected 0.0)" << std::endl;
    assert(std::abs(d_self) < 1e-6f);

    // Distance between u and v
    float d_uv = weft::subriemannian_distance(u, v);
    std::cout << "[TEST] Sub-Riemannian Distance d_CC(u, v) = " << d_uv << std::endl;
    assert(d_uv > 0.0f);

    // Test Carnot Dilation Homogeneity: delta_lambda(u) = (lambda*x, lambda*y, lambda^2*z)
    const float lambda = 2.0f;
    weft::HeisenbergPoint u_dilated(lambda * u.x, lambda * u.y, lambda * lambda * u.z);
    weft::HeisenbergPoint v_dilated(lambda * v.x, lambda * v.y, lambda * lambda * v.z);

    float d_dilated = weft::subriemannian_distance(u_dilated, v_dilated);
    std::cout << "[TEST] Dilated Distance d_CC(delta_lambda u, delta_lambda v) = " << d_dilated
              << " (Expected " << lambda * d_uv << ")" << std::endl;

    float rel_diff = std::abs(d_dilated - lambda * d_uv) / (lambda * d_uv);
    assert(rel_diff < 1e-5f);
    std::cout << "[TEST] Carnot Dilation Homogeneity Verified." << std::endl;

    std::cout << "[TEST] All Sub-Riemannian Carnot Group Attention tests passed successfully!" << std::endl;
    return 0;
}
