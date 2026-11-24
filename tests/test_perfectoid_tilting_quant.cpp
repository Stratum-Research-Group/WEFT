// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Perfectoid Tilting Quantization Invariants (Paper 213)

#include "csrc/cli/weft_perfectoid_tilting_quant.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_tilt_untilt_cycle() {
    const size_t N = 32;
    std::vector<float> orig(N);
    std::vector<float> tilted(N, 0.0f);
    std::vector<float> scales(N, 2.0f);
    std::vector<float> recon(N, 0.0f);

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) orig[i] = 1.0f;
        else if (i % 3 == 1) orig[i] = -1.0f;
        else orig[i] = 0.0f;
    }

    weft::perfectoid_tilting::tilt_weights_avx512(orig.data(), tilted.data(), N, 1.0f);

    for (size_t i = 0; i < N; ++i) {
        if (i % 3 == 0) assert(tilted[i] == 1.0f);
        else if (i % 3 == 1) assert(tilted[i] == -1.0f);
        else assert(tilted[i] == 0.0f);
    }

    weft::perfectoid_tilting::untilt_weights_avx512(tilted.data(), scales.data(), recon.data(), N);

    for (size_t i = 0; i < N; ++i) {
        assert(std::abs(recon[i] - tilted[i] * 2.0f) < 1e-6f);
    }
    std::cout << "[PASS] test_tilt_untilt_cycle (N=32 verified)" << std::endl;
}

int main() {
    std::cout << "=== Running Paper 213: Perfectoid Tilting Unit Tests ===" << std::endl;
    test_tilt_untilt_cycle();
    std::cout << "All Perfectoid Tilting tests PASSED." << std::endl;
    return 0;
}
