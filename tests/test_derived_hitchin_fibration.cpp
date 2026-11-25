// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_derived_hitchin_fibration.h"

int main() {
    std::cout << "[TEST] Running Hitchin Fibration Eviction Engine AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> k_trivial(dim, 0.05f);
    std::vector<float> k_critical(dim, 1.5f);

    float energy_trivial = weft::HitchinFibrationEvictionEngine::compute_spectral_energy(k_trivial.data(), dim);
    float energy_critical = weft::HitchinFibrationEvictionEngine::compute_spectral_energy(k_critical.data(), dim);

    assert(std::abs(energy_trivial - (0.05f * 0.05f)) < 1e-5f);
    assert(std::abs(energy_critical - (1.5f * 1.5f)) < 1e-5f);
    assert(weft::HitchinFibrationEvictionEngine::is_token_evictable(k_trivial.data(), dim, 0.1f));
    assert(!weft::HitchinFibrationEvictionEngine::is_token_evictable(k_critical.data(), dim, 0.1f));

    std::cout << "[PASS] Hitchin Fibration Eviction Engine verified successfully." << std::endl;
    return 0;
}
