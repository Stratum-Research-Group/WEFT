#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_logarithmic_speculation.h"

int main() {
    std::cout << "[TEST] Running Logarithmic Geometry Speculative Verification Invariant Test..." << std::endl;

    const size_t VOCAB_SIZE = 32;
    std::vector<float> target_logits(VOCAB_SIZE, 2.0f);

    // Test 1: Identical logits -> 0 log residual, verified
    std::vector<float> perfect_draft = target_logits;
    weft::LogResidualProfile prof0 = weft::verify_speculative_candidate_logarithmic(
        perfect_draft.data(), target_logits.data(), VOCAB_SIZE, 0.1f
    );
    std::cout << "  Perfect draft max log residual: " << prof0.max_log_residual << std::endl;
    assert(prof0.max_log_residual == 0.0f);
    assert(prof0.is_verified);

    // Test 2: Bounded divergence within threshold
    std::vector<float> good_draft = target_logits;
    good_draft[0] = 2.05f; // residual = 0.05 / 2.0 = 0.025 <= 0.1
    weft::LogResidualProfile prof1 = weft::verify_speculative_candidate_logarithmic(
        good_draft.data(), target_logits.data(), VOCAB_SIZE, 0.1f
    );
    std::cout << "  Good draft max log residual: " << prof1.max_log_residual << std::endl;
    assert(prof1.is_verified);
    assert(prof1.max_log_residual > 0.0f);

    // Test 3: Large divergence exceeds threshold -> rejected
    std::vector<float> bad_draft = target_logits;
    bad_draft[0] = 3.5f; // residual = 1.5 / 2.0 = 0.75 > 0.1
    weft::LogResidualProfile prof2 = weft::verify_speculative_candidate_logarithmic(
        bad_draft.data(), target_logits.data(), VOCAB_SIZE, 0.1f
    );
    std::cout << "  Bad draft max log residual: " << prof2.max_log_residual << std::endl;
    assert(!prof2.is_verified);
    assert(prof2.max_log_residual > 0.1f);

    std::cout << "[PASS] All Logarithmic Geometry Speculative Verification invariants verified." << std::endl;
    return 0;
}
