// test_finsler_attention.cpp
// Validation of non-reversible Randers Finsler distance and forward triangle inequality.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_finsler_attention.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_finsler_attention..." << std::endl;

    const int dim = 32;
    std::vector<float> u(dim, 0.0f);
    std::vector<float> v(dim, 1.0f);
    std::vector<float> b(dim, 0.05f); // Sub-critical drift: ||b||^2 = 32 * 0.0025 = 0.08 < 1.0

    // Self distance F(u, u) = 0
    float d_self = weft::randers_distance_avx512(u.data(), u.data(), b.data(), dim);
    std::cout << "[TEST] Randers Self Distance F(u, u) = " << d_self << " (Expected 0.0)" << std::endl;
    assert(std::abs(d_self) < 1e-6f);

    // Forward distance F(u, v) vs Backward distance F(v, u)
    float d_fwd = weft::randers_distance_avx512(u.data(), v.data(), b.data(), dim);
    float d_bwd = weft::randers_distance_avx512(v.data(), u.data(), b.data(), dim);

    std::cout << "[TEST] Forward Distance F(u, v) = " << d_fwd << std::endl;
    std::cout << "[TEST] Backward Distance F(v, u) = " << d_bwd << std::endl;

    // Strict asymmetry: F(u, v) != F(v, u)
    assert(d_fwd != d_bwd);
    assert(d_fwd > 0.0f && d_bwd > 0.0f);
    std::cout << "[TEST] Strict Metric Asymmetry Confirmed." << std::endl;

    // Check forward triangle inequality: F(u, w) <= F(u, v) + F(v, w)
    std::vector<float> w(dim, 2.5f);
    float d_uw = weft::randers_distance_avx512(u.data(), w.data(), b.data(), dim);
    float d_vw = weft::randers_distance_avx512(v.data(), w.data(), b.data(), dim);

    std::cout << "[TEST] Triangle Check: F(u, w)=" << d_uw << " <= F(u, v)+F(v, w)=" << d_fwd + d_vw << std::endl;
    assert(d_uw <= d_fwd + d_vw + 1e-5f);

    std::cout << "[TEST] All Finsler Geometric Attention tests passed successfully!" << std::endl;
    return 0;
}
