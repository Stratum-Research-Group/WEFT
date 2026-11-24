// test_derived_intersection.cpp
// Milestone Paper 231: Verification of AVX-512 Derived Intersection & Virtual Fundamental Classes
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_derived_intersection.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_config_and_invariants() {
    weft::DerivedIntersectionConfig cfg;
    cfg.ambient_dim = 128;
    cfg.subvariety_dim_x = 96;
    cfg.subvariety_dim_y = 64;
    cfg.actual_dim = 48;

    // Expected vdim = 96 + 64 - 128 = 32
    assert(cfg.virtual_dim() == 32);

    // Factor = 32 / 48 = 0.666667
    assert(std::fabs(cfg.virtual_class_factor() - (2.0f / 3.0f)) < 1e-5f);

    // Transverse case: actual == virtual
    cfg.actual_dim = 32;
    assert(std::fabs(cfg.virtual_class_factor() - 1.0f) < 1e-5f);
}

void test_avx512_projection_kernel() {
    weft::DerivedIntersectionConfig cfg{128, 96, 64, 48};
    float factor = cfg.virtual_class_factor();

    const size_t N = 64;
    std::vector<float> query(N, 1.5f);
    std::vector<float> key(N, 2.0f);
    std::vector<float> out(N, 0.0f);

    weft::avx512_derived_intersection_project(query.data(), key.data(), out.data(), N, cfg);

    float expected = 1.5f * 2.0f * factor;
    for (size_t i = 0; i < N; ++i) {
        assert(std::fabs(out[i] - expected) < 1e-5f);
    }
}

int main() {
    std::cout << "[Paper 231] Running Derived Intersection & Virtual Fundamental Class Unit Tests...\n";
    test_config_and_invariants();
    test_avx512_projection_kernel();
    std::cout << "[Paper 231] All unit tests PASSED successfully.\n";
    return 0;
}
