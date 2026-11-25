// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include "../csrc/cli/weft_tropical_toric_ideals.h"

int main() {
    std::cout << "[TEST] Running Tropical Bit Slicing Engine AVX-512 Test..." << std::endl;

    const size_t num_channels = 32;
    std::vector<float> variances(num_channels);
    std::vector<uint8_t> precisions(num_channels, 0);

    for (size_t i = 0; i < num_channels; ++i) {
        if (i < 10) variances[i] = 0.05f; // low variance -> 1 bit
        else if (i < 20) variances[i] = 0.5f; // mid variance -> 2 bit
        else variances[i] = 2.0f; // high variance -> 4 bit
    }

    weft::TropicalBitSlicingEngine::schedule_bit_precisions(
        variances.data(), precisions.data(), num_channels, 0.2f, 1.0f
    );

    for (size_t i = 0; i < num_channels; ++i) {
        uint8_t expected = (i < 10) ? 1 : ((i < 20) ? 2 : 4);
        assert(precisions[i] == expected);
    }

    std::cout << "[PASS] Tropical Bit Slicing Engine verified successfully." << std::endl;
    return 0;
}
