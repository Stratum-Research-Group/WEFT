// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_fused_rmsnorm_dual_residual.h"

int main() {
    std::cout << "[TEST] Running Fused RMSNorm Dual Residual Engine AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> x(dim);
    std::vector<float> res1(dim);
    std::vector<float> res2(dim);
    std::vector<float> gamma(dim, 1.0f);
    std::vector<float> out(dim, 0.0f);

    for (size_t i = 0; i < dim; ++i) {
        x[i] = 1.0f + 0.05f * (i % 5);
        res1[i] = 0.5f - 0.02f * (i % 7);
        res2[i] = 0.25f + 0.01f * (i % 3);
        gamma[i] = 1.0f + 0.1f * (i % 4);
    }

    weft::FusedRMSNormDualResidualEngine::evaluate_norm(
        x.data(), res1.data(), res2.data(), gamma.data(), out.data(), dim
    );

    float sq_sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float s = x[i] + res1[i] + res2[i];
        sq_sum += s * s;
    }
    float rms = 1.0f / std::sqrt((sq_sum / (float)dim) + 1e-6f);

    for (size_t i = 0; i < dim; ++i) {
        float s = x[i] + res1[i] + res2[i];
        float expected = (s * rms) * gamma[i];
        assert(std::abs(out[i] - expected) < 1e-5f);
    }

    std::cout << "[PASS] Fused RMSNorm Dual Residual Engine verified successfully." << std::endl;
    return 0;
}
