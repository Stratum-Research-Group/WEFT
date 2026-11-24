// test_hyperbolic_kv.cpp
// Numerical validation of Lorentz Hyperboloid geometry and reversed Cauchy-Schwarz bound.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_hyperbolic_kv.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_hyperbolic_kv..." << std::endl;

    const int d = 64; // Spatial dimension
    std::vector<float> u_spatial(d, 0.0f);
    std::vector<float> v_spatial(d, 0.0f);

    // Point u: Origin of hyperboloid (0, 0, ..., 0), time u0 = 1.0
    float u0 = weft::compute_lorentz_time_component(u_spatial.data(), d);
    std::cout << "[TEST] Origin Time Component u0: " << u0 << " (Expected 1.0)" << std::endl;
    assert(std::abs(u0 - 1.0f) < 1e-6f);

    // Self distance d_H(u, u) = 0
    float dist_self = weft::hyperbolic_geodesic_distance(u_spatial.data(), u0, u_spatial.data(), u0, d);
    std::cout << "[TEST] Geodesic Distance d_H(u, u): " << dist_self << " (Expected 0.0)" << std::endl;
    assert(std::abs(dist_self) < 1e-5f);

    // Point v: Displaced point with spatial components
    for (int i = 0; i < d; ++i) {
        v_spatial[i] = 0.1f;
    }
    float v0 = weft::compute_lorentz_time_component(v_spatial.data(), d);
    std::cout << "[TEST] Displaced Point Time Component v0: " << v0 << std::endl;
    assert(v0 > 1.0f);

    // Check Reversed Cauchy-Schwarz: -<u, v>_L >= 1
    float inner_prod = weft::minkowski_inner_product_avx512(u_spatial.data(), u0, v_spatial.data(), v0, d);
    std::cout << "[TEST] Lorentzian Inner Product <u, v>_L: " << inner_prod << std::endl;
    assert(-inner_prod >= 1.0f);

    float dist_uv = weft::hyperbolic_geodesic_distance(u_spatial.data(), u0, v_spatial.data(), v0, d);
    std::cout << "[TEST] Geodesic Distance d_H(u, v): " << dist_uv << std::endl;
    assert(dist_uv > 0.0f);

    // Check Symmetry: d_H(u, v) == d_H(v, u)
    float dist_vu = weft::hyperbolic_geodesic_distance(v_spatial.data(), v0, u_spatial.data(), u0, d);
    assert(std::abs(dist_uv - dist_vu) < 1e-5f);

    std::cout << "[TEST] All Hyperbolic Riemannian KV tests passed successfully!" << std::endl;
    return 0;
}
