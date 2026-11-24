// test_lattice_kv_quantization.cpp - Verification for Paper #131
#include "csrc/cli/weft_lattice_kv.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace weft;

int main() {
    std::cout << "[TEST] Initializing test_lattice_kv_quantization..." << std::endl;

    // Test 1: Test Conway-Sloane D4 Projection
    // Vector with natural even parity sum: [1.1, 1.9, 0.2, 0.8] -> round: [1, 2, 0, 1] -> sum = 4 (even)
    float in_even[4] = {1.1f, 1.9f, 0.2f, 0.8f};
    int8_t out_even[4];
    WeftLatticeKVEngine::project_to_d4(in_even, out_even);

    int sum_even = out_even[0] + out_even[1] + out_even[2] + out_even[3];
    std::cout << "[TEST] Even parity D4 projected: ["
              << (int)out_even[0] << ", " << (int)out_even[1] << ", "
              << (int)out_even[2] << ", " << (int)out_even[3] << "] | Sum = " << sum_even << std::endl;
    assert(std::abs(sum_even) % 2 == 0);

    // Vector with odd parity sum: [1.1, 2.1, 0.1, 0.1] -> round: [1, 2, 0, 0] -> sum = 3 (odd)
    // Largest residual is 1.1 - 1.0 = 0.1, or 2.1 - 2.0 = 0.1; flipping one restores even parity
    float in_odd[4] = {1.1f, 2.1f, 0.1f, 0.1f};
    int8_t out_odd[4];
    WeftLatticeKVEngine::project_to_d4(in_odd, out_odd);

    int sum_odd = out_odd[0] + out_odd[1] + out_odd[2] + out_odd[3];
    std::cout << "[TEST] Odd parity corrected D4 projected: ["
              << (int)out_odd[0] << ", " << (int)out_odd[1] << ", "
              << (int)out_odd[2] << ", " << (int)out_odd[3] << "] | Sum = " << sum_odd << std::endl;
    assert(std::abs(sum_odd) % 2 == 0);

    // Test 2: In-Register Lattice Dot Product
    std::vector<float> query(WeftLatticeKVEngine::HEAD_DIM, 1.0f);
    WeftLatticeKVEngine::LatticeTokenBlock block;
    block.scale = 0.2f;
    block.zero_point = 0.0f;
    for (size_t b = 0; b < WeftLatticeKVEngine::NUM_BLOCKS; ++b) {
        // Packed 4 coordinates of value 2: 0b10101010
        block.packed_d4[b] = 0b10101010;
    }

    float dot = WeftLatticeKVEngine::compute_lattice_dot_product(query.data(), block);
    // 64 * 1.0 * (2 * 0.2) / sqrt(64) = 64 * 0.4 / 8 = 25.6 / 8 = 3.2
    std::cout << "[TEST] Computed Lattice Dot Product: " << dot << " (Expected 3.2)" << std::endl;
    assert(std::abs(dot - 3.2f) < 1e-5f);

    std::cout << "[TEST] All D4 coordinate lattice compaction checks passed successfully!" << std::endl;
    return 0;
}
