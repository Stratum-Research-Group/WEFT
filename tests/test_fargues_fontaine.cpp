#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_fargues_fontaine.h"

int main() {
    std::cout << "[TEST] Running Fargues-Fontaine Curves & Vector Bundles Invariant Test..." << std::endl;

    const size_t NUM_TOKENS = 64;
    std::vector<float> token_energies(NUM_TOKENS, 0.0f);

    // Tokens 0..15 have high degree (positive slope)
    for (size_t t = 0; t < 16; ++t) {
        token_energies[t] = 2.5f;
    }
    // Tokens 16..63 have negative degree/slope
    for (size_t t = 16; t < NUM_TOKENS; ++t) {
        token_energies[t] = 0.2f;
    }

    float threshold = 1.0f; // threshold = 1.0 -> 0..15 have slope > 0, 16..63 have slope < 0
    weft::FFContextProfile prof = weft::filter_fargues_fontaine_context(
        token_energies.data(), NUM_TOKENS, threshold
    );

    std::cout << "  Total tokens: " << prof.total_tokens << std::endl;
    std::cout << "  Retained tokens: " << prof.retained_tokens << " (expected: 16)" << std::endl;
    std::cout << "  Evicted tokens: " << prof.evicted_tokens << " (expected: 48)" << std::endl;
    std::cout << "  Compression ratio: " << prof.slope_compression_ratio << "x" << std::endl;

    assert(prof.retained_tokens == 16);
    assert(prof.evicted_tokens == 48);
    assert(prof.slope_compression_ratio == 4.0f);

    std::cout << "[PASS] All Fargues-Fontaine Curves & Vector Bundles invariants verified." << std::endl;
    return 0;
}
