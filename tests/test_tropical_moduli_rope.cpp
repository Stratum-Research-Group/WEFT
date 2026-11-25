// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_tropical_moduli_rope.h"

int main() {
    std::cout << "[TEST] Running Tropical RoPE Engine AVX-512 Test..." << std::endl;

    const size_t half_dim = 32;
    std::vector<float> x(half_dim * 2);
    std::vector<float> cos_tab(half_dim);
    std::vector<float> sin_tab(half_dim);
    std::vector<float> x_out(half_dim * 2, 0.0f);

    for (size_t i = 0; i < half_dim; ++i) {
        x[i] = 1.0f + 0.05f * (i % 5);
        x[half_dim + i] = 0.5f + 0.02f * (i % 7);
        float angle = 0.1f * (float)(i + 1);
        cos_tab[i] = std::cos(angle);
        sin_tab[i] = std::sin(angle);
    }

    weft::TropicalRoPEEngine::apply_rope(
        x.data(), cos_tab.data(), sin_tab.data(), x_out.data(), half_dim
    );

    // Verify norm preservation: ||x_out||^2 == ||x||^2
    float norm_in = 0.0f;
    float norm_out = 0.0f;
    for (size_t i = 0; i < half_dim; ++i) {
        float in_sq = x[i] * x[i] + x[half_dim + i] * x[half_dim + i];
        float out_sq = x_out[i] * x_out[i] + x_out[half_dim + i] * x_out[half_dim + i];
        norm_in += in_sq;
        norm_out += out_sq;
        assert(std::abs(in_sq - out_sq) < 1e-5f);
    }

    assert(std::abs(norm_in - norm_out) < 1e-4f);
    std::cout << "[PASS] Tropical RoPE Engine verified successfully (Norms: in=" << norm_in << ", out=" << norm_out << ")." << std::endl;
    return 0;
}
