#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_local_langlands.h"

int main() {
    std::cout << "[TEST] Running Geometrization of Local Langlands Attention Invariant Test..." << std::endl;

    const size_t NUM_HEADS = 4;
    const size_t HEAD_DIM = 16;
    std::vector<float> heads(NUM_HEADS * HEAD_DIM, 0.0f);

    // Initialize 4 perfectly orthogonal heads
    for (size_t h = 0; h < NUM_HEADS; ++h) {
        for (size_t d = h * 4; d < (h + 1) * 4; ++d) {
            heads[h * HEAD_DIM + d] = 1.0f;
        }
    }

    weft::LanglandsSliceProfile prof = weft::compute_langlands_spectral_slices(
        heads.data(), NUM_HEADS, HEAD_DIM
    );

    std::cout << "  Num Heads: " << prof.num_heads << std::endl;
    std::cout << "  Max Cross-Correlation: " << prof.max_cross_correlation << " (expected 0.0)" << std::endl;
    std::cout << "  Total Spectral Energy: " << prof.total_spectral_energy << " (expected 16.0)" << std::endl;

    assert(prof.num_heads == 4);
    assert(prof.max_cross_correlation < 1e-5f);
    assert(std::abs(prof.total_spectral_energy - 16.0f) < 1e-5f);

    std::cout << "[PASS] All Geometrization of Local Langlands Attention invariants verified." << std::endl;
    return 0;
}
