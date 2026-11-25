// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_fused_swiglu.h"

int main() {
    std::cout << "[TEST] Running Fused SwiGLU Engine AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> gate(dim);
    std::vector<float> up(dim);
    std::vector<float> out(dim, 0.0f);

    for (size_t i = 0; i < dim; ++i) {
        gate[i] = -2.0f + 0.1f * (i % 40);
        up[i] = 1.0f + 0.05f * (i % 8);
    }

    weft::FusedSwiGLUEngine::evaluate_swiglu(gate.data(), up.data(), out.data(), dim);

    for (size_t i = 0; i < dim; ++i) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + std::exp(-g));
        float exp_val = (g * sig) * up[i];
        assert(std::abs(out[i] - exp_val) < 1e-5f);
    }

    std::cout << "[PASS] Fused SwiGLU Engine verified successfully." << std::endl;
    return 0;
}
