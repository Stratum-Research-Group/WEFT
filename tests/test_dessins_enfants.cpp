// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Dessins d'Enfants Attention Permutation Invariants (Paper 210)

#include "csrc/cli/weft_dessins_enfants.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <numeric>

void test_sigma_infinity_monodromy() {
    // 4-edge cycle: sigma_0 = (0 1 2 3), sigma_1 = (0)(1 2)(3)
    std::vector<uint32_t> s0 = {1, 2, 3, 0};
    std::vector<uint32_t> s1 = {0, 2, 1, 3};

    std::vector<uint32_t> s_inf = weft::dessins_enfants::compute_sigma_infinity(s0, s1);

    // Verify s_inf * s0 * s1 = identity: s_inf[s0[s1[i]]] == i
    for (size_t i = 0; i < 4; ++i) {
        uint32_t comp = s_inf[s0[s1[i]]];
        assert(comp == i);
    }
    std::cout << "[PASS] test_sigma_infinity_monodromy (relation verified)" << std::endl;
}

void test_permute_channels_gather() {
    const size_t N = 32;
    std::vector<float> src(N);
    std::vector<float> dst(N, 0.0f);
    std::vector<uint32_t> perm(N);

    for (size_t i = 0; i < N; ++i) {
        src[i] = static_cast<float>(i * 10);
        perm[i] = static_cast<uint32_t>((i + 3) % N);
    }

    weft::dessins_enfants::permute_dessin_channels(src.data(), perm.data(), dst.data(), N);

    for (size_t i = 0; i < N; ++i) {
        float expected = src[perm[i]];
        assert(dst[i] == expected);
    }
    std::cout << "[PASS] test_permute_channels_gather (N=32 verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 210: Dessins d'Enfants Unit Tests ===" << std::endl;
    test_sigma_infinity_monodromy();
    test_permute_channels_gather();
    std::cout << "All Dessins d'Enfants tests PASSED." << std::endl;
    return 0;
}
