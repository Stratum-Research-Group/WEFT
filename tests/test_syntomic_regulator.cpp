// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Syntomic Regulator Draft Horizon Invariants (Paper 228)

#include "csrc/cli/weft_syntomic_regulator.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_compute_optimal_draft_horizon() {
    const size_t max_K = 8;
    // p = 0.5 each step -> -log2(0.5) = 1.0 vol per step
    std::vector<float> probs(max_K, 0.5f);

    // Tau = 2.5 -> reaches >= 2.5 at step 3 (accum = 3.0)
    size_t k_star = weft::syntomic_regulator::compute_optimal_draft_horizon_avx512(
        probs.data(), max_K, 2.5f
    );
    assert(k_star == 3);
    std::cout << "[PASS] test_compute_optimal_draft_horizon (K* = 3 verified)" << std::endl;
}

void test_batch_compute_horizons() {
    const size_t branches = 4;
    const size_t max_K = 6;
    std::vector<float> all_probs(branches * max_K, 0.5f);
    std::vector<uint32_t> out_k(branches, 0);

    weft::syntomic_regulator::batch_compute_horizons_avx512(
        all_probs.data(), branches, max_K, out_k.data(), 2.0f
    );

    // Tau = 2.0 -> reached at step 2 (1.0 + 1.0 = 2.0)
    for (size_t b = 0; b < branches; ++b) {
        assert(out_k[b] == 2);
    }
    std::cout << "[PASS] test_batch_compute_horizons (all branches K*=2 verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 228: Syntomic Regulators Unit Tests ===" << std::endl;
    test_compute_optimal_draft_horizon();
    test_batch_compute_horizons();
    std::cout << "All Syntomic Regulators tests PASSED." << std::endl;
    return 0;
}
