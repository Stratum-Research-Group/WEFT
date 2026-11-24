// test_voronoi_quant.cpp
// Validation of Conway-Sloane E8 Gosset lattice Voronoi quantization and covering radius bound.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_voronoi_quant.h"
#include <iostream>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_voronoi_quant..." << std::endl;

    // Test 1: Exact lattice point in Z^8 with even sum: (2, 0, 0, 0, 0, 0, 0, 0)
    float x1[8] = {2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float lambda1[8];
    weft::quantize_e8_conway_sloane_avx512(x1, lambda1);

    for (int i = 0; i < 8; ++i) {
        assert(std::abs(x1[i] - lambda1[i]) < 1e-6f);
    }
    std::cout << "[TEST] Exact Lattice Point Decoding Confirmed." << std::endl;

    // Test 2: Arbitrary vector inside covering radius (R^2 <= 1.0)
    float x2[8] = {0.3f, -0.4f, 0.2f, 0.1f, -0.2f, 0.5f, -0.1f, 0.2f};
    float lambda2[8];
    weft::quantize_e8_conway_sloane_avx512(x2, lambda2);

    float err_sq = 0.0f;
    float sum_coords = 0.0f;
    for (int i = 0; i < 8; ++i) {
        float d = x2[i] - lambda2[i];
        err_sq += d * d;
        sum_coords += lambda2[i];
    }
    std::cout << "[TEST] E8 Quantization Squared Error: " << err_sq << " (Covering Radius Bound: <= 1.0)" << std::endl;
    assert(err_sq <= 1.0f + 1e-5f);

    // Verify even coordinate sum parity condition in E8
    int int_sum = static_cast<int>(std::round(sum_coords));
    std::cout << "[TEST] Quantized E8 Coordinate Sum: " << sum_coords << " (Parity Mod 2: " << (int_sum % 2) << ")" << std::endl;
    assert(int_sum % 2 == 0);

    std::cout << "[TEST] All Voronoi Delaunay Quantization tests passed successfully!" << std::endl;
    return 0;
}
