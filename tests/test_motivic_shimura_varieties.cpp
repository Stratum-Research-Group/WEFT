// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_motivic_shimura_varieties.h"

int main() {
    std::cout << "[TEST] Running Shimura Hecke Synthesis Engine AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> v1(dim);
    std::vector<float> v2(dim);
    std::vector<float> out(dim, 0.0f);

    float alpha1 = 0.6f;
    float alpha2 = 0.8f;

    for (size_t i = 0; i < dim; ++i) {
        v1[i] = 1.0f + 0.05f * (i % 6);
        v2[i] = 2.0f - 0.02f * (i % 8);
    }

    weft::ShimuraHeckeSynthesisEngine::project_hecke_2subwords(
        v1.data(), alpha1, v2.data(), alpha2, out.data(), dim
    );

    for (size_t i = 0; i < dim; ++i) {
        float expected = alpha1 * v1[i] + alpha2 * v2[i];
        assert(std::abs(out[i] - expected) < 1e-5f);
    }

    std::cout << "[PASS] Shimura Hecke Synthesis Engine verified successfully." << std::endl;
    return 0;
}
