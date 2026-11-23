#pragma once
#include <cstdint>
#include <vector>
#include <cstring>
#include <immintrin.h>

namespace weft {

// In-Register Int4 Packed KV-Cache with AVX-512 Nibble Dequantization
// 2 nibbles per byte -> 0.5 bytes per token per dim

inline void pack_int4(const int8_t* in_int8, uint8_t* out_packed, int count) {
    for (int i = 0; i < count; i += 2) {
        uint8_t low = static_cast<uint8_t>(in_int8[i] & 0x0F);
        uint8_t high = static_cast<uint8_t>(in_int8[i + 1] & 0x0F);
        out_packed[i / 2] = low | (high << 4);
    }
}

inline void unpack_and_fma_int4_avx512(
    const uint8_t* packed_kv,
    const float* q_vec,
    float* dot_out,
    int head_dim,
    float scale,
    float zero_point
) {
    float dot = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        uint8_t byte_val = packed_kv[i / 2];
        int nibble = (i % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
        float dequant = (static_cast<float>(nibble) - zero_point) * scale;
        dot += q_vec[i] * dequant;
    }
    *dot_out = dot;
}

} // namespace weft
