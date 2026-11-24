// test_padic_kv.cpp
// Validation of 2-adic valuation and strong ultrametric triangle inequality.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_padic_kv.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_padic_kv..." << std::endl;

    // Check valuations:
    // v_2(8) = 3 (8 = 2^3)
    // v_2(12) = 2 (12 = 4 * 3 = 2^2 * 3)
    assert(weft::v2_valuation(8) == 3);
    assert(weft::v2_valuation(12) == 2);
    assert(weft::v2_valuation(7) == 0); // Odd number -> 2^0
    std::cout << "[TEST] 2-adic Valuations Verified." << std::endl;

    // Test strong ultrametric triangle inequality across diverse triplets (x, y, z)
    uint32_t test_triplets[5][3] = {
        {4, 12, 28},
        {1, 5, 9},
        {100, 200, 300},
        {7, 15, 31},
        {1024, 2048, 4096}
    };

    for (int i = 0; i < 5; ++i) {
        uint32_t x = test_triplets[i][0];
        uint32_t y = test_triplets[i][1];
        uint32_t z = test_triplets[i][2];

        bool satisfies = weft::check_ultrametric_property(x, y, z);
        float d_xz = weft::padic_distance_2(x, z);
        float d_xy = weft::padic_distance_2(x, y);
        float d_yz = weft::padic_distance_2(y, z);

        std::cout << "[TEST] Triplet (" << x << ", " << y << ", " << z << "): d(x,z)="
                  << d_xz << " <= max(" << d_xy << ", " << d_yz << ")" << std::endl;
        assert(satisfies);
    }

    std::cout << "[TEST] All p-Adic Ultrametric KV tests passed successfully!" << std::endl;
    return 0;
}
