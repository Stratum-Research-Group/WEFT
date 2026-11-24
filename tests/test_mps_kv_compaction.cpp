// test_mps_kv_compaction.cpp
// Validation of continuous-variable MPS Schmidt singular value truncation and trace re-normalization.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_mps_kv_compaction.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_mps_kv_compaction..." << std::endl;

    const int chi_in = 16;
    const int chi_max = 4;
    std::vector<float> lambdas(chi_in);

    // Exponentially decaying Schmidt spectrum
    float sum_sq = 0.0f;
    for (int i = 0; i < chi_in; ++i) {
        lambdas[i] = std::exp(-0.5f * float(i));
        sum_sq += lambdas[i] * lambdas[i];
    }
    // Normalize initial state
    for (int i = 0; i < chi_in; ++i) {
        lambdas[i] /= std::sqrt(sum_sq);
    }

    float entropy_initial = weft::compute_entanglement_entropy(lambdas.data(), chi_in);
    std::cout << "[TEST] Initial Entanglement Entropy S_vN: " << entropy_initial << std::endl;
    assert(entropy_initial > 0.0f);

    std::vector<float> truncated(chi_max);
    int chi_out = weft::truncate_and_renormalize_mps_avx512(lambdas.data(), chi_in, truncated.data(), chi_max);
    assert(chi_out == chi_max);

    // Verify trace normalization: sum(truncated_i^2) == 1.0
    float trunc_sum_sq = 0.0f;
    for (int i = 0; i < chi_max; ++i) {
        trunc_sum_sq += truncated[i] * truncated[i];
    }
    std::cout << "[TEST] Post-Truncation Normalized Trace: " << trunc_sum_sq << std::endl;
    assert(std::abs(trunc_sum_sq - 1.0f) < 1e-6f);

    float entropy_truncated = weft::compute_entanglement_entropy(truncated.data(), chi_max);
    std::cout << "[TEST] Truncated Entanglement Entropy S_vN: " << entropy_truncated << std::endl;
    assert(entropy_truncated <= entropy_initial);

    std::cout << "[TEST] All Continuous-Variable MPS Compaction tests passed successfully!" << std::endl;
    return 0;
}
