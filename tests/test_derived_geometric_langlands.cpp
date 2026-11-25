// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_derived_geometric_langlands.h"

int main() {
    std::cout << "[TEST] Running Geometric Langlands Alignment Engine AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> a(dim);
    std::vector<float> b(dim);
    std::vector<float> out(dim, 0.0f);

    float b_norm_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        a[i] = 1.0f + 0.05f * (i % 6);
        b[i] = 0.1f * ((i % 4) + 1);
        b_norm_sq += b[i] * b[i];
    }

    // Normalize basis b
    float inv_norm_b = 1.0f / std::sqrt(b_norm_sq);
    for (size_t i = 0; i < dim; ++i) {
        b[i] *= inv_norm_b;
    }

    weft::GeometricLanglandsAlignmentEngine::project_eigensheaf(
        a.data(), b.data(), out.data(), dim
    );

    float dot_expected = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot_expected += a[i] * b[i];
    }

    for (size_t i = 0; i < dim; ++i) {
        float expected = dot_expected * b[i];
        assert(std::abs(out[i] - expected) < 1e-5f);
    }

    std::cout << "[PASS] Geometric Langlands Alignment Engine verified successfully." << std::endl;
    return 0;
}
