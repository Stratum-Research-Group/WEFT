// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_fused_ternary_softmax.h"

int main() {
    std::cout << "[TEST] Running Fused Ternary Softmax Engine AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> x(dim);
    std::vector<float> out(dim, 0.0f);

    for (size_t i = 0; i < dim; ++i) {
        x[i] = 0.1f * ((i % 10) - 5);
    }

    weft::FusedTernarySoftmaxEngine::evaluate_softmax(x.data(), out.data(), dim);

    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += out[i];
        assert(out[i] >= 0.0f);
    }

    assert(std::abs(sum - 1.0f) < 1e-4f);
    std::cout << "[PASS] Fused Ternary Softmax Engine verified successfully (sum: " << sum << ")." << std::endl;
    return 0;
}
