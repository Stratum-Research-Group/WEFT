// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_tropical_toric_degenerations.h"

int main() {
    std::cout << "[TEST] Running Tropical Toric Degenerations AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> k_a(dim);
    std::vector<float> k_b(dim);
    std::vector<float> k_out(dim);

    for (size_t i = 0; i < dim; ++i) {
        k_a[i] = 1.0f + 0.05f * (i % 7);
        k_b[i] = k_a[i] + ((i == 42) ? 0.35f : 0.02f);
    }

    float dist = weft::TropicalToricCompactor::compute_tropical_distance(k_a.data(), k_b.data(), dim);
    assert(std::abs(dist - 0.35f) < 1e-5f);

    weft::TropicalToricCompactor::merge_tropical_keys(k_a.data(), k_b.data(), k_out.data(), dim);
    for (size_t i = 0; i < dim; ++i) {
        float expected = 0.5f * (k_a[i] + k_b[i]);
        assert(std::abs(k_out[i] - expected) < 1e-5f);
    }

    std::cout << "[PASS] Tropical Toric Degenerations kernel verified successfully." << std::endl;
    return 0;
}
