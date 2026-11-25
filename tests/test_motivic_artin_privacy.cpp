// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_motivic_artin_privacy.h"

int main() {
    std::cout << "[TEST] Running Artin Stack Privacy Engine AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> a(dim);
    std::vector<float> mask_A(dim, 0.0f);
    std::vector<float> mask_B(dim, 0.0f);
    std::vector<float> a_out_A(dim, 0.0f);

    for (size_t i = 0; i < dim; ++i) {
        a[i] = 2.5f + 0.1f * (i % 7);
        if (i < dim / 2) {
            mask_A[i] = 1.0f; // Tenant A occupies first half
        } else {
            mask_B[i] = 1.0f; // Tenant B occupies second half
        }
    }

    weft::ArtinStackPrivacyEngine::apply_tenant_mask(a.data(), mask_A.data(), a_out_A.data(), dim);
    float leakage = weft::ArtinStackPrivacyEngine::compute_cross_tenant_interference(
        a_out_A.data(), mask_B.data(), dim
    );

    assert(leakage < 1e-6f);
    std::cout << "[PASS] Artin Stack Privacy Engine verified successfully (cross leakage: " << leakage << ")." << std::endl;
    return 0;
}
