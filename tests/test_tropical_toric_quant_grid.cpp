// test_tropical_toric_quant_grid.cpp
// Milestone Paper 241: Verification of AVX-512 Tropical Toric Quantization Grids
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_tropical_toric_quant_grid.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_dequantize_grid_kernel() {
    const size_t N = 16;
    std::vector<float> q_levels(N);
    for (size_t i = 0; i < N; ++i) {
        q_levels[i] = static_cast<float>(i);
    }
    std::vector<float> out(N, 0.0f);

    weft::ToricQuantConfig cfg{0.1f, 0.02f};
    weft::avx512_toric_dequantize_grid(q_levels.data(), out.data(), N, cfg);

    // Verify center symmetry: level 7 (-0.5) vs level 8 (+0.5)
    assert(std::fabs(out[7] + out[8]) < 1e-5f);

    // Verify strict monotonicity
    for (size_t i = 0; i + 1 < N; ++i) {
        assert(out[i] < out[i+1]);
    }
}

int main() {
    std::cout << "[Paper 241] Running Tropical Toric Quantization Grid Unit Tests...\n";
    test_dequantize_grid_kernel();
    std::cout << "[Paper 241] All unit tests PASSED successfully.\n";
    return 0;
}
