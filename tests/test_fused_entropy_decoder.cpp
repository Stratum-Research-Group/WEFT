#include <iostream>
#include <vector>
#include <cassert>
#include "csrc/cli/weft_fused_entropy_decoder.h"

int main() {
    std::cout << "[TEST] Running test_fused_entropy_decoder..." << std::endl;

    // Stream: 0, 1, -1, 0 -> binary: 00 (0), 01 (+1), 10 (-1), 00 (0)
    // Low-to-high shifts: 00 | (01 << 2) | (10 << 4) | (00 << 6) = 0x00 | 0x04 | 0x20 | 0x00 = 0x24
    uint8_t compressed[1] = {0x24};
    int8_t decompressed[4] = {0};

    weft::FusedEntropyDecoder::decode_2bit_huffman_stream(compressed, decompressed, 4);

    assert(decompressed[0] == 0);
    assert(decompressed[1] == 1);
    assert(decompressed[2] == -1);
    assert(decompressed[3] == 0);
    std::cout << "  Decoded 4 ternary values correctly: "
              << (int)decompressed[0] << ", "
              << (int)decompressed[1] << ", "
              << (int)decompressed[2] << ", "
              << (int)decompressed[3] << std::endl;

    // Test Elias-Fano low-bits extraction
    // Store 4-bit numbers: 5, 11, 3 in 64-bit stream
    uint64_t low_stream[2] = {0};
    low_stream[0] = 5ULL | (11ULL << 4) | (3ULL << 8);
    uint32_t indices[3] = {0};
    weft::FusedEntropyDecoder::decode_elias_fano_indices(low_stream, 4, indices, 3);
    assert(indices[0] == 5);
    assert(indices[1] == 11);
    assert(indices[2] == 3);
    std::cout << "  Elias-Fano indices decoded: " << indices[0] << ", " << indices[1] << ", " << indices[2] << std::endl;

    std::cout << "[PASS] test_fused_entropy_decoder succeeded." << std::endl;
    return 0;
}
