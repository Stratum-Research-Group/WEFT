// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_motivic_enriques_surfaces.h"

int main() {
    std::cout << "[TEST] Running Enriques Attestation Engine AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> a(dim);
    std::vector<float> secret(dim);

    for (size_t i = 0; i < dim; ++i) {
        a[i] = 1.0f + 0.05f * (i % 8);
        secret[i] = 0.5f - 0.02f * (i % 6);
    }

    float score = weft::EnriquesAttestationEngine::compute_attestation_score(a.data(), secret.data(), dim);
    assert(weft::EnriquesAttestationEngine::verify_attestation(a.data(), secret.data(), dim, score));

    // Perturb activation
    a[0] += 0.5f;
    assert(!weft::EnriquesAttestationEngine::verify_attestation(a.data(), secret.data(), dim, score));

    std::cout << "[PASS] Enriques Attestation Engine verified successfully (score: " << score << ")." << std::endl;
    return 0;
}
