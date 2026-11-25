// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_derived_calabi_yau_pruning.h"

int main() {
    std::cout << "[TEST] Running Calabi-Yau Mirror Pruning Engine AVX-512 Test..." << std::endl;

    const size_t dim = 128;
    std::vector<float> w_sparse(dim, 0.0f);
    std::vector<float> w_dense(dim, 0.0f);

    for (size_t i = 0; i < dim; ++i) {
        if (i % 8 == 0) w_sparse[i] = 1.0f; // 12.5% active
        w_dense[i] = ((i % 2) == 0) ? 1.0f : -1.0f; // 100% active
    }

    float energy_sparse = weft::CalabiYauMirrorPruningEngine::compute_mirror_energy(w_sparse.data(), dim);
    float energy_dense = weft::CalabiYauMirrorPruningEngine::compute_mirror_energy(w_dense.data(), dim);

    assert(std::abs(energy_sparse - 0.125f) < 1e-5f);
    assert(std::abs(energy_dense - 1.0f) < 1e-5f);
    assert(weft::CalabiYauMirrorPruningEngine::is_row_prunable(w_sparse.data(), dim, 0.2f));
    assert(!weft::CalabiYauMirrorPruningEngine::is_row_prunable(w_dense.data(), dim, 0.2f));

    std::cout << "[PASS] Calabi-Yau Mirror Pruning Engine verified successfully." << std::endl;
    return 0;
}
