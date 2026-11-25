// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_tropical_flag_varieties.h"

int main() {
    std::cout << "[TEST] Running Tropical Flag Low-Rank Engine AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    const size_t rank_r = 4;
    std::vector<float> k(dim);
    std::vector<float> U(rank_r * dim, 0.0f);
    std::vector<float> c(rank_r, 0.0f);
    std::vector<float> k_recon(dim, 0.0f);

    for (size_t i = 0; i < dim; ++i) {
        k[i] = 1.0f + 0.05f * (i % 8);
    }

    // Set orthonormal rows in U (for first 4 coordinates)
    for (size_t r = 0; r < rank_r; ++r) {
        U[r * dim + r] = 1.0f;
    }

    weft::TropicalFlagLowRankEngine::project_low_rank(k.data(), U.data(), c.data(), rank_r, dim);
    for (size_t r = 0; r < rank_r; ++r) {
        assert(std::abs(c[r] - k[r]) < 1e-5f);
    }

    weft::TropicalFlagLowRankEngine::reconstruct_key(c.data(), U.data(), k_recon.data(), rank_r, dim);
    for (size_t r = 0; r < rank_r; ++r) {
        assert(std::abs(k_recon[r] - k[r]) < 1e-5f);
    }
    for (size_t i = rank_r; i < dim; ++i) {
        assert(std::abs(k_recon[i] - 0.0f) < 1e-5f);
    }

    std::cout << "[PASS] Tropical Flag Low-Rank Engine verified successfully." << std::endl;
    return 0;
}
