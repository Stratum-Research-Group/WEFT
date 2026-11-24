#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_grand_unified_engine.h"

int main() {
    std::cout << "[TEST] Running Centennial Grand Unified Arithmetic Engine Invariant Test..." << std::endl;

    const size_t NUM_TOKENS = 64;
    const size_t DIM = 16;
    std::vector<float> features(NUM_TOKENS * DIM, 0.2f);

    // Active high-slope tokens (tokens 0..15)
    for (size_t t = 0; t < 16; ++t) {
        for (size_t d = 0; d < DIM; ++d) {
            features[t * DIM + d] = 1.0f;
        }
    }

    float adic_thresh = 1.5f;
    float slope_thresh = 0.5f;

    weft::GrandUnifiedReport rep = weft::execute_grand_unified_inference_step(
        features.data(), NUM_TOKENS, DIM, adic_thresh, slope_thresh
    );

    std::cout << "  Processed tokens: " << rep.processed_tokens << std::endl;
    std::cout << "  Retained KV bundles: " << rep.retained_kv_bundles << " (expected 16)" << std::endl;
    std::cout << "  Max adic valuation: " << rep.max_adic_valuation << " (expected 1.0)" << std::endl;
    std::cout << "  Quaternionic stability norm: " << rep.quaternionic_stability_norm << std::endl;
    std::cout << "  All invariants satisfied: " << (rep.all_invariants_satisfied ? "true" : "false") << std::endl;

    assert(rep.processed_tokens == NUM_TOKENS);
    assert(rep.retained_kv_bundles == 16);
    assert(std::abs(rep.max_adic_valuation - 1.0f) < 1e-5f);
    assert(rep.all_invariants_satisfied);
    assert(rep.quaternionic_stability_norm > 0.0f);

    std::cout << "[PASS] All Centennial Grand Unified Arithmetic Engine invariants verified." << std::endl;
    return 0;
}
