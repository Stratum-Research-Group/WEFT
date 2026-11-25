// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_tropical_toric_fibers.h"

int main() {
    std::cout << "[TEST] Running Tropical Toric Fibers Skip Engine AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> hl(dim);
    std::vector<float> hl1_close(dim);
    std::vector<float> hl1_far(dim);

    for (size_t i = 0; i < dim; ++i) {
        hl[i] = 1.0f + 0.05f * (i % 8);
        hl1_close[i] = hl[i] + 0.005f; // small change -> skippable
        hl1_far[i] = hl[i] + 0.8f;     // large change -> not skippable
    }

    float div_close = weft::TropicalFiberSkipEngine::compute_layer_divergence(hl.data(), hl1_close.data(), dim);
    float div_far = weft::TropicalFiberSkipEngine::compute_layer_divergence(hl.data(), hl1_far.data(), dim);

    assert(std::abs(div_close - 0.005f) < 1e-5f);
    assert(std::abs(div_far - 0.8f) < 1e-5f);
    assert(weft::TropicalFiberSkipEngine::should_skip_layer(hl.data(), hl1_close.data(), dim, 0.01f));
    assert(!weft::TropicalFiberSkipEngine::should_skip_layer(hl.data(), hl1_far.data(), dim, 0.01f));

    std::cout << "[PASS] Tropical Toric Fibers Skip Engine verified successfully." << std::endl;
    return 0;
}
