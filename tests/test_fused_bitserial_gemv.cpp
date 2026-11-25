#include "csrc/cli/weft_fused_bitserial_gemv.h"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    constexpr size_t num_blocks = 4; // 256 weights
    std::vector<uint64_t> pos_bits(num_blocks, 0);
    std::vector<uint64_t> neg_bits(num_blocks, 0);
    std::vector<uint64_t> act_bits(num_blocks, 0xFFFFFFFFFFFFFFFFULL); // All active

    // Set 10 positive bits and 3 negative bits in block 0
    pos_bits[0] = 0x3FFULL; // 10 bits set
    neg_bits[0] = 0x1C00ULL; // 3 bits set

    float dot = weft::fused_bitserial_gemv_avx512(
        pos_bits.data(), neg_bits.data(), act_bits.data(), num_blocks, 1.0f);

    // Expected dot = 10 - 3 = 7.0f
    assert(dot == 7.0f);

    std::cout << "[PASS] Fused Bit-Serial Popcount GEMV Unit Test passed (Dot: " << dot << ").\n";
    return 0;
}
