// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Tropical Toric Amoeba Sparsity Invariants (Paper 211)

#include "csrc/cli/weft_tropical_amoeba.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_tropical_norm_properties() {
    weft::tropical_amoeba::TropicalCoordinate p1{3.0f, -5.0f};
    float norm1 = weft::tropical_amoeba::compute_tropical_norm(p1);
    assert(std::abs(norm1 - 5.0f) < 1e-6f);

    weft::tropical_amoeba::TropicalCoordinate p2{0.0f, 0.0f};
    assert(weft::tropical_amoeba::compute_tropical_norm(p2) == 0.0f);
    std::cout << "[PASS] test_tropical_norm_properties" << std::endl;
}

void test_filter_amoeba_avx512() {
    const size_t N = 64;
    std::vector<float> v1(N);
    std::vector<float> v2(N);
    std::vector<uint32_t> retained(N, 0);

    for (size_t i = 0; i < N; ++i) {
        v1[i] = static_cast<float>(i) * 0.1f;
        v2[i] = static_cast<float>(i % 4);
    }

    float tau = 2.5f;
    size_t count = weft::tropical_amoeba::filter_tokens_amoeba_avx512(
        v1.data(), v2.data(), N, tau, retained.data()
    );

    // Verify all retained elements have max(|v1|, |v2|) <= tau
    for (size_t k = 0; k < count; ++k) {
        uint32_t idx = retained[k];
        float norm = std::max(std::abs(v1[idx]), std::abs(v2[idx]));
        assert(norm <= tau);
    }
    std::cout << "[PASS] test_filter_amoeba_avx512 (retained " << count << " of " << N << ")" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 211: Tropical Toric Amoeba Unit Tests ===" << std::endl;
    test_tropical_norm_properties();
    test_filter_amoeba_avx512();
    std::cout << "All Tropical Toric Amoeba tests PASSED." << std::endl;
    return 0;
}
