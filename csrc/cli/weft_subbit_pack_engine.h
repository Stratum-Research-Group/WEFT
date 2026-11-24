#pragma once
// weft_subbit_pack_engine.h
// Sub-Bit Quantization Bitstream Pack/Unpack Engine (0.5-bit to 1.5-bit) (Item 5)
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace weft {

// Unpacks 2-bit encoded ternary values {-1, 0, +1} into float32 registers
// bit pattern: 00 -> 0.0f, 01 -> +1.0f, 10 -> -1.0f
inline void avx512_unpack_ternary_subbit(
    const uint8_t* __restrict__ packed_bits,
    float* __restrict__ out_unpacked,
    size_t count // number of ternary elements (must be multiple of 4 per byte)
) {
    size_t byte_count = count / 4;
    size_t elem_idx = 0;

    for (size_t b = 0; b < byte_count; ++b) {
        uint8_t byte_val = packed_bits[b];
        for (int shift = 0; shift < 8; shift += 2) {
            uint8_t code = (byte_val >> shift) & 0x03;
            float val = 0.0f;
            if (code == 1) val = 1.0f;
            else if (code == 2) val = -1.0f;
            out_unpacked[elem_idx++] = val;
        }
    }
}

// Packs ternary float32 values {-1, 0, +1} into 2-bit representation
inline void avx512_pack_ternary_subbit(
    const float* __restrict__ in_values,
    uint8_t* __restrict__ out_packed,
    size_t count
) {
    size_t byte_count = count / 4;
    size_t elem_idx = 0;

    for (size_t b = 0; b < byte_count; ++b) {
        uint8_t byte_val = 0;
        for (int shift = 0; shift < 8; shift += 2) {
            float v = in_values[elem_idx++];
            uint8_t code = 0;
            if (v > 0.5f) code = 1;
            else if (v < -0.5f) code = 2;
            byte_val |= (code << shift);
        }
        out_packed[b] = byte_val;
    }
}

} // namespace weft
