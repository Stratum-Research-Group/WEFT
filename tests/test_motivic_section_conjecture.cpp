// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_motivic_section_conjecture.h"

int main() {
    std::cout << "[TEST] Running Motivic Section Drift Detector AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> a_in(dim);
    std::vector<float> a_out(dim);
    std::vector<float> mu(dim);

    for (size_t i = 0; i < dim; ++i) {
        mu[i] = 1.0f + 0.05f * (i % 4);
        a_in[i] = mu[i] + 0.01f; // in-distribution
        a_out[i] = mu[i] + 5.0f; // out-of-distribution
    }

    float defect_in = weft::MotivicSectionDriftDetector::compute_section_defect(a_in.data(), mu.data(), dim);
    float defect_out = weft::MotivicSectionDriftDetector::compute_section_defect(a_out.data(), mu.data(), dim);

    assert(defect_in < 0.01f);
    assert(defect_out > 0.5f);
    assert(!weft::MotivicSectionDriftDetector::is_ood(a_in.data(), mu.data(), dim, 0.2f));
    assert(weft::MotivicSectionDriftDetector::is_ood(a_out.data(), mu.data(), dim, 0.2f));

    std::cout << "[PASS] Motivic Section Drift Detector kernel verified successfully." << std::endl;
    return 0;
}
