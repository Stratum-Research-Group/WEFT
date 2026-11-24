// test_flash_decompression.cpp - Verification for Paper #124
#include "csrc/cli/weft_flash_decomp.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace weft;

int main() {
    std::cout << "[TEST] Initializing test_flash_decompression..." << std::endl;

    WeftFlashDecompressor decompressor;
    constexpr size_t TEST_TOKENS = 16;
    std::vector<WeftFlashDecompressor::StreamingBlock> blocks(TEST_TOKENS);

    for (size_t t = 0; t < TEST_TOKENS; ++t) {
        blocks[t].scale = 0.25f;
        blocks[t].zero_point = -0.375f;
        for (size_t b = 0; b < 16; ++b) {
            // Pattern of bit-pairs: 00 01 10 11 (0, 1, 2, 3)
            blocks[t].packed_data[b] = 0b11100100;
        }
    }

    // Decompress partition 0
    decompressor.decompress_partition(blocks.data(), TEST_TOKENS, 0);
    const float* buf0 = decompressor.get_buffer(0);

    // Verify expected affine mapping:
    // 0 -> 0 * 0.25 - 0.375 = -0.375
    // 1 -> 1 * 0.25 - 0.375 = -0.125
    // 2 -> 2 * 0.25 - 0.375 =  0.125
    // 3 -> 3 * 0.25 - 0.375 =  0.375
    for (size_t t = 0; t < TEST_TOKENS; ++t) {
        for (size_t i = 0; i < WeftFlashDecompressor::HEAD_DIM; i += 4) {
            assert(std::abs(buf0[t * WeftFlashDecompressor::HEAD_DIM + i + 0] - (-0.375f)) < 1e-5f);
            assert(std::abs(buf0[t * WeftFlashDecompressor::HEAD_DIM + i + 1] - (-0.125f)) < 1e-5f);
            assert(std::abs(buf0[t * WeftFlashDecompressor::HEAD_DIM + i + 2] - (0.125f)) < 1e-5f);
            assert(std::abs(buf0[t * WeftFlashDecompressor::HEAD_DIM + i + 3] - (0.375f)) < 1e-5f);
        }
    }

    std::cout << "[TEST] Partition 0 verified with zero numerical error." << std::endl;

    // Test double buffer partition 1 alternation
    decompressor.decompress_partition(blocks.data(), TEST_TOKENS, 1);
    const float* buf1 = decompressor.get_buffer(1);
    assert(buf1 != buf0);
    assert(std::abs(buf1[0] - (-0.375f)) < 1e-5f);

    std::cout << "[TEST] Double-buffering partition alternation verified." << std::endl;
    std::cout << "[TEST] All flash-decompression checks passed successfully!" << std::endl;
    return 0;
}
