// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Higher Chow Groups & Motivic Spectral Sequences (Paper 209)

#include "csrc/cli/weft_higher_chow.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_boundary_nilpotence() {
    weft::higher_chow::ChowCycle c{2, 1, 10.0f};
    weft::higher_chow::ChowCycle dc = weft::higher_chow::boundary_diff(c);
    weft::higher_chow::ChowCycle d2c = weft::higher_chow::boundary_diff(dc);

    assert(dc.degree == 0);
    assert(d2c.degree == 0);
    assert(d2c.energy == 0.0f);
    std::cout << "[PASS] test_boundary_nilpotence (d^2 degree 0, energy 0)" << std::endl;
}

void test_layer_fusion_convex() {
    const size_t N = 64;
    std::vector<float> layer_a(N, 1.0f);
    std::vector<float> layer_b(N, 3.0f);
    std::vector<float> merged(N, 0.0f);

    // Alpha = 0.25, Beta = 0.75 -> expected = 1.0*0.25 + 3.0*0.75 = 0.25 + 2.25 = 2.50
    weft::higher_chow::fuse_layers_chow(
        layer_a.data(), layer_b.data(),
        1.0f, 3.0f,
        merged.data(), N
    );

    for (size_t i = 0; i < N; ++i) {
        assert(std::abs(merged[i] - 2.5f) < 1e-5f);
    }
    std::cout << "[PASS] test_layer_fusion_convex (N=64, merged value 2.50)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 209: Higher Chow Groups Unit Tests ===" << std::endl;
    test_boundary_nilpotence();
    test_layer_fusion_convex();
    std::cout << "All Higher Chow Groups tests PASSED." << std::endl;
    return 0;
}
