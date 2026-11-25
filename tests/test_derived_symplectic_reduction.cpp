// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_derived_symplectic_reduction.h"

int main() {
    std::cout << "[TEST] Running Symplectic Reduction Engine AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> v(dim);
    std::vector<float> w(dim);
    std::vector<float> v_out(dim);

    for (size_t i = 0; i < dim; ++i) {
        v[i] = 1.0f + 0.1f * (i % 6);
        w[i] = 2.0f - 0.05f * (i % 8);
    }

    weft::SymplecticReductionEngine::project_orthogonal(v.data(), w.data(), v_out.data(), dim);
    float residual = weft::SymplecticReductionEngine::compute_residual_dot(v_out.data(), w.data(), dim);

    assert(residual < 1e-4f);
    std::cout << "[PASS] Symplectic Reduction Engine verified successfully (residual: " << residual << ")." << std::endl;
    return 0;
}
