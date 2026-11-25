// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_tropical_toric_divisors.h"

int main() {
    std::cout << "[TEST] Running Tropical Toric Divisors Activation AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> x(dim);
    std::vector<float> out(dim, 0.0f);

    for (size_t i = 0; i < dim; ++i) {
        x[i] = -4.0f + 0.1f * (float)i; // spans [-4.0, +8.7]
    }

    weft::TropicalPiecewiseActivationEngine::evaluate_pwl_activation(x.data(), out.data(), dim);

    for (size_t i = 0; i < dim; ++i) {
        float val = x[i];
        float expected = 0.0f;
        if (val >= 2.0f) expected = val;
        else if (val >= -2.0f) expected = 0.25f * (val + 2.0f);
        else expected = 0.0f;

        assert(std::abs(out[i] - expected) < 1e-5f);
    }

    std::cout << "[PASS] Tropical Toric Divisors Activation verified successfully." << std::endl;
    return 0;
}
