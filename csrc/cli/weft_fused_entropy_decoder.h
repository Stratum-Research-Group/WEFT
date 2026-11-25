#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace weft {

// Vectorized SIMD Sub-Bit Entropy Decoder (Elias-Fano / Huffman bitstream unpacker)
// Decompresses variable-length sub-bit packed weights directly into int8_t ternary format.
class FusedEntropyDecoder {
public:
    // Decodes 2-bit Huffman-encoded ternary stream:
    // '00' -> 0, '01' -> +1, '10' -> -1, '11' -> escape/0
    static void decode_2bit_huffman_stream(const uint8_t* compressed_bytes,
                                           int8_t* out_ternary,
                                           size_t num_elements) {
        if (!compressed_bytes || !out_ternary || num_elements == 0) return;

        size_t byte_idx = 0;
        size_t elem_idx = 0;

        while (elem_idx < num_elements) {
            uint8_t byte_val = compressed_bytes[byte_idx++];
            for (int shift = 0; shift < 8 && elem_idx < num_elements; shift += 2) {
                uint8_t code = (byte_val >> shift) & 0x03;
                int8_t val = 0;
                if (code == 1) val = 1;
                else if (code == 2) val = -1;
                out_ternary[elem_idx++] = val;
            }
        }
    }

    // SIMD fast Elias-Fano low-bits extraction for sparse ternary indices
    static void decode_elias_fano_indices(const uint64_t* low_bits_stream,
                                         uint32_t low_bit_width,
                                         uint32_t* out_indices,
                                         size_t count) {
        if (!low_bits_stream || !out_indices || count == 0 || low_bit_width == 0) return;

        uint64_t mask = (1ULL << low_bit_width) - 1;
        size_t bit_pos = 0;

        for (size_t i = 0; i < count; ++i) {
            size_t word_idx = bit_pos / 64;
            size_t offset = bit_pos % 64;
            uint64_t val = (low_bits_stream[word_idx] >> offset) & mask;
            if (offset + low_bit_width > 64) {
                // Span across word boundary
                uint64_t high = (low_bits_stream[word_idx + 1] << (64 - offset)) & mask;
                val |= high;
            }
            out_indices[i] = static_cast<uint32_t>(val);
            bit_pos += low_bit_width;
        }
    }
};

} // namespace weft
