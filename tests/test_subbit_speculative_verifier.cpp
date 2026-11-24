#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_subbit_speculative_verifier.h"

int main() {
    std::cout << "[Test] Initializing Sub-Bit Speculative Verifier SIMD Test..." << std::endl;

    const size_t num_candidates = 4;
    const size_t vocab_dim = 64;
    const float threshold = 0.95f;

    std::vector<float> target_logits(num_candidates * vocab_dim, 1.0f);
    std::vector<float> draft_logits(num_candidates * vocab_dim, 1.0f);
    std::vector<float> out_residuals(num_candidates, 0.0f);

    // Make candidate 0 and 1 match perfectly
    for (size_t i = 0; i < 2 * vocab_dim; ++i) {
        target_logits[i] = static_cast<float>(i % 11 + 1);
        draft_logits[i] = static_cast<float>(i % 11 + 1);
    }

    // Candidate 2 deviates strongly
    for (size_t i = 2 * vocab_dim; i < 3 * vocab_dim; ++i) {
        target_logits[i] = 10.0f;
        draft_logits[i] = -10.0f;
    }

    size_t accepted = weft::avx512_verify_speculative_candidates(
        target_logits.data(),
        draft_logits.data(),
        out_residuals.data(),
        num_candidates,
        vocab_dim,
        threshold
    );

    std::cout << "[Test] Accepted Candidates: " << accepted << " / " << num_candidates << std::endl;
    assert(accepted == 2);
    assert(std::fabs(out_residuals[0] - 1.0f) < 1e-4f);
    assert(std::fabs(out_residuals[1] - 1.0f) < 1e-4f);
    assert(out_residuals[2] < 0.0f);

    std::cout << "[Test] Sub-Bit Speculative Verifier SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
