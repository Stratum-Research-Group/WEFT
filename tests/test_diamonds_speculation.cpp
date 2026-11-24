#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_diamonds_speculation.h"

int main() {
    std::cout << "[TEST] Running Diamonds & v-Topologies Speculative Branching Invariant Test..." << std::endl;

    const size_t NUM_CANDIDATES = 4;
    const size_t FEATURE_DIM = 16;
    std::vector<float> target_verifications(NUM_CANDIDATES * FEATURE_DIM, 1.0f);
    std::vector<float> draft_projections(NUM_CANDIDATES * FEATURE_DIM, 1.0f);

    // Candidates 0, 1, 2 agree with target (residual = 0.0 <= 0.1)
    // Candidate 3 diverges (residual = 2.0 > 0.1)
    for (size_t d = 0; d < FEATURE_DIM; ++d) {
        draft_projections[3 * FEATURE_DIM + d] = 1.5f; // diff = 0.5 * sqrt(16) = 2.0
    }

    float threshold = 0.1f;
    weft::DiamondBranchProfile prof = weft::verify_diamond_speculative_branches(
        draft_projections.data(), target_verifications.data(), NUM_CANDIDATES, FEATURE_DIM, threshold
    );

    std::cout << "  Num candidates: " << prof.num_candidates << std::endl;
    std::cout << "  Accepted candidates: " << prof.accepted_candidates << " (expected: 3)" << std::endl;
    std::cout << "  Max cocycle residual: " << prof.max_cocycle_residual << " (expected: 2.0)" << std::endl;
    std::cout << "  Is v-descent satisfied: " << (prof.is_v_descent_satisfied ? "true" : "false") << std::endl;

    assert(prof.num_candidates == 4);
    assert(prof.accepted_candidates == 3);
    assert(std::abs(prof.max_cocycle_residual - 2.0f) < 1e-4f);
    assert(!prof.is_v_descent_satisfied);

    std::cout << "[PASS] All Diamonds & v-Topologies Speculative Branching invariants verified." << std::endl;
    return 0;
}
