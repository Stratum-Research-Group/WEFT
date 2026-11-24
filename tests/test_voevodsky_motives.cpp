#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_voevodsky_motives.h"

int main() {
    std::cout << "[TEST] Running Voevodsky Motives Weight Quantization Test..." << std::endl;

    const size_t DIM = 32;
    std::vector<float> w_in(DIM, 0.5f);
    std::vector<float> w_out(DIM, 0.0f);

    // Test 1: In-register motivic Tate projection: c0=0.8, c1=0.2
    // w = 0.5 -> w^3 = 0.125
    // w_out = 0.8 * 0.5 + 0.2 * 0.125 = 0.4 + 0.025 = 0.425
    weft::project_voevodsky_motives_simd(w_in.data(), DIM, 0.8f, 0.2f, w_out.data());

    for (size_t i = 0; i < DIM; ++i) {
        assert(std::abs(w_out[i] - 0.425f) < 1e-5f);
    }
    std::cout << "  Projected motivic weight: " << w_out[0] << " (Expected: 0.425)" << std::endl;

    // Test 2: Invariance of zero weight
    std::vector<float> zero_in(DIM, 0.0f);
    std::vector<float> zero_out(DIM, 1.0f);
    weft::project_voevodsky_motives_simd(zero_in.data(), DIM, 0.8f, 0.2f, zero_out.data());
    for (size_t i = 0; i < DIM; ++i) {
        assert(zero_out[i] == 0.0f);
    }
    std::cout << "  Zero weight invariant preserved." << std::endl;

    std::cout << "[PASS] All Voevodsky Motives Quantization invariants verified." << std::endl;
    return 0;
}
