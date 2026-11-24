// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Čerednik-Drinfeld p-Adic Uniformization Invariants (Paper 208)

#include "csrc/cli/weft_cerednik_drinfeld.h"
#include <iostream>
#include <cassert>
#include <vector>

void test_tree_distance_properties() {
    weft::cerednik_drinfeld::TreeVertex u{4, 1};
    weft::cerednik_drinfeld::TreeVertex v{7, 1}; // same component, diff levels
    weft::cerednik_drinfeld::TreeVertex w{7, 2}; // diff component, same level as v

    // Self distance is 0
    assert(weft::cerednik_drinfeld::compute_tree_distance(u, u) == 0);

    // Symmetry
    uint32_t d_uv = weft::cerednik_drinfeld::compute_tree_distance(u, v);
    uint32_t d_vu = weft::cerednik_drinfeld::compute_tree_distance(v, u);
    assert(d_uv == d_vu);
    assert(d_uv == 3); // 7 - 4 = 3

    // Cross component penalty
    uint32_t d_vw = weft::cerednik_drinfeld::compute_tree_distance(v, w);
    assert(d_vw == 2); // 0 level diff + 2 component diff

    std::cout << "[PASS] test_tree_distance_properties" << std::endl;
}

void test_nearest_codebook_entry() {
    std::vector<weft::cerednik_drinfeld::TreeVertex> codebook = {
        {1, 0},
        {5, 0},
        {10, 1},
        {20, 2}
    };
    weft::cerednik_drinfeld::TreeVertex query{6, 0};
    size_t nearest = weft::cerednik_drinfeld::find_nearest_codebook_entry(query, codebook);
    assert(nearest == 1); // {5, 0} has distance |6-5| + 0 = 1
    std::cout << "[PASS] test_nearest_codebook_entry (index: " << nearest << ")" << std::endl;
}

void test_batch_distance_simd() {
    const size_t N = 32;
    uint32_t q_l = 8;
    uint32_t q_c = 1;
    std::vector<uint32_t> levels(N);
    std::vector<uint32_t> comps(N);
    std::vector<uint32_t> out(N, 0);

    for (size_t i = 0; i < N; ++i) {
        levels[i] = static_cast<uint32_t>(i % 16);
        comps[i] = (i % 2 == 0) ? 1 : 0;
    }

    weft::cerednik_drinfeld::batch_tree_distance_avx512(
        q_l, q_c,
        levels.data(), comps.data(),
        out.data(), N
    );

    for (size_t i = 0; i < N; ++i) {
        weft::cerednik_drinfeld::TreeVertex q{q_l, q_c};
        weft::cerednik_drinfeld::TreeVertex cb{levels[i], comps[i]};
        uint32_t expected = weft::cerednik_drinfeld::compute_tree_distance(q, cb);
        assert(out[i] == expected);
    }
    std::cout << "[PASS] test_batch_distance_simd (N=32 verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 208: Čerednik-Drinfeld Unit Tests ===" << std::endl;
    test_tree_distance_properties();
    test_nearest_codebook_entry();
    test_batch_distance_simd();
    std::cout << "All Čerednik-Drinfeld tests PASSED." << std::endl;
    return 0;
}
