// test_orthogonal_draft_generation.cpp - Verification for Paper #132
#include "csrc/cli/weft_orthogonal_draft.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace weft;

int main() {
    std::cout << "[TEST] Initializing test_orthogonal_draft_generation..." << std::endl;

    WeftOrthogonalDraftEngine engine(/*collinear_threshold=*/1e-4f);

    constexpr size_t NUM_INPUT_BRANCHES = 4;
    std::vector<float> in_branches(NUM_INPUT_BRANCHES * WeftOrthogonalDraftEngine::LATENT_DIM, 0.0f);

    // Branch 0: Unit vector along first dimension
    in_branches[0 * WeftOrthogonalDraftEngine::LATENT_DIM + 0] = 1.0f;

    // Branch 1: Orthogonal unit vector along second dimension
    in_branches[1 * WeftOrthogonalDraftEngine::LATENT_DIM + 1] = 1.0f;

    // Branch 2: Collinear with Branch 0 (e.g. 2.0 * Branch 0) -> Should be pruned!
    in_branches[2 * WeftOrthogonalDraftEngine::LATENT_DIM + 0] = 2.0f;

    // Branch 3: Linearly independent mix (1.0 along dim 0, 1.0 along dim 2) -> Residual along dim 2 should be retained
    in_branches[3 * WeftOrthogonalDraftEngine::LATENT_DIM + 0] = 1.0f;
    in_branches[3 * WeftOrthogonalDraftEngine::LATENT_DIM + 2] = 1.0f;

    std::vector<float> out_basis(WeftOrthogonalDraftEngine::MAX_BRANCHES * WeftOrthogonalDraftEngine::LATENT_DIM, 0.0f);

    size_t accepted = engine.orthogonalize_draft_branches(
        in_branches.data(),
        NUM_INPUT_BRANCHES,
        out_basis.data()
    );

    std::cout << "[TEST] Input branches: " << NUM_INPUT_BRANCHES << " | Accepted orthogonal branches: " << accepted << std::endl;
    // Branch 2 was collinear, so only 3 branches should be accepted
    assert(accepted == 3);

    // Verify mutual orthogonality: dot(u_i, u_j) == 0 for all i != j
    for (size_t i = 0; i < accepted; ++i) {
        for (size_t j = i + 1; j < accepted; ++j) {
            float dot = 0.0f;
            for (size_t d = 0; d < WeftOrthogonalDraftEngine::LATENT_DIM; ++d) {
                dot += out_basis[i * WeftOrthogonalDraftEngine::LATENT_DIM + d] *
                       out_basis[j * WeftOrthogonalDraftEngine::LATENT_DIM + d];
            }
            std::cout << "[TEST] dot(basis[" << i << "], basis[" << j << "]) = " << dot << std::endl;
            assert(std::abs(dot) < 1e-5f);
        }
    }

    std::cout << "[TEST] All orthogonal Gram-Schmidt draft pruning checks passed successfully!" << std::endl;
    return 0;
}
