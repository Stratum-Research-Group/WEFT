#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_higher_topos.h"

int main() {
    std::cout << "[TEST] Running Higher Topos Theory & ∞-Category Descent Invariant Test..." << std::endl;

    const size_t NUM_TOKENS = 16;
    const size_t FEATURE_DIM = 32;
    std::vector<float> features(NUM_TOKENS * FEATURE_DIM, 1.0f);

    size_t trunc_level = 16; // Truncate at level 16
    weft::InfinitySheafProfile prof = weft::aggregate_infinity_topos_cochains(
        features.data(), NUM_TOKENS, FEATURE_DIM, trunc_level
    );

    std::cout << "  Truncated Level: " << prof.truncated_level << std::endl;
    std::cout << "  Semantic Coherence Norm: " << prof.semantic_coherence_norm << std::endl;
    std::cout << "  Descent Ratio per Token: " << prof.hypercomplete_descent_ratio << " (expected: 4.0)" << std::endl;

    assert(prof.truncated_level == 16);
    assert(std::abs(prof.hypercomplete_descent_ratio - 4.0f) < 1e-5f);
    assert(prof.semantic_coherence_norm > 0.0f);

    std::cout << "[PASS] All Higher Topos Theory & ∞-Category Descent invariants verified." << std::endl;
    return 0;
}
