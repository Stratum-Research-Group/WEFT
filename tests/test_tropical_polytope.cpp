// test_tropical_polytope.cpp
// Validation of tropical convex hull lower bounds and branch pruning decisions.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_tropical_polytope.h"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_tropical_polytope..." << std::endl;

    const int num_candidates = 4;
    const int dim = 16;
    std::vector<float> candidates(num_candidates * dim);
    std::vector<float> weights(dim, 1.0f);

    // Populate candidates with positive log-penalties
    for (int r = 0; r < num_candidates; ++r) {
        for (int c = 0; c < dim; ++c) {
            candidates[r * dim + c] = 5.0f + float(r + c);
        }
    }
    // Min cost in candidates: r=0, c=0 -> 5.0 + 1.0 = 6.0

    // Test pruning with threshold tau = 4.0: all costs >= 6.0 > 4.0 -> PRUNE
    bool prune_high = weft::should_prune_tropical_branch_avx512(candidates.data(), weights.data(), num_candidates, dim, 4.0f);
    std::cout << "[TEST] Prune Decision (tau=4.0): " << (prune_high ? "PRUNE" : "KEEP") << " (Expected PRUNE)" << std::endl;
    assert(prune_high == true);

    // Test pruning with threshold tau = 8.0: min cost 6.0 <= 8.0 -> KEEP
    bool prune_low = weft::should_prune_tropical_branch_avx512(candidates.data(), weights.data(), num_candidates, dim, 8.0f);
    std::cout << "[TEST] Prune Decision (tau=8.0): " << (prune_low ? "PRUNE" : "KEEP") << " (Expected KEEP)" << std::endl;
    assert(prune_low == false);

    // Test tropical convex combination: alpha=0, beta=1
    std::vector<float> x(dim, 2.0f);
    std::vector<float> y(dim, 4.0f);
    std::vector<float> z(dim);
    weft::tropical_convex_combination(x.data(), y.data(), z.data(), dim, 0.0f, 1.0f);
    // z_j = min(0 + 2, 1 + 4) = min(2, 5) = 2.0
    for (int j = 0; j < dim; ++j) {
        assert(z[j] == 2.0f);
    }
    std::cout << "[TEST] Tropical Convex Combination Verified." << std::endl;

    std::cout << "[TEST] All Tropical Polytope Speculative Bounding tests passed successfully!" << std::endl;
    return 0;
}
