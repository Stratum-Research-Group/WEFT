#include "csrc/cli/weft_derived_cinfinity_rings.h"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    constexpr size_t dim = 64;
    std::vector<float> root_grad(dim, 1.0f);
    std::vector<float> candidate_jets(16 * dim, 0.0f);
    float scores[16] = {0.0f};

    // Candidate 3 has strong positive alignment (0.1f * 64 = 6.4f)
    for (size_t d = 0; d < dim; ++d) {
        candidate_jets[3 * dim + d] = 0.1f;
    }

    weft::synthetic_draft_verify_16candidates_avx512(
        root_grad.data(), candidate_jets.data(), scores, dim, 0.5f);

    assert(scores[3] > 6.0f);
    assert(scores[0] == 0.0f);

    std::cout << "[PASS] Derived C-Infinity Rings Synthetic Draft Verifier Unit Test passed (Candidate 3 score: " 
              << scores[3] << ").\n";
    return 0;
}
