#ifndef WEFT_FUSED_INT4_TERNARY_GEMV_H
#define WEFT_FUSED_INT4_TERNARY_GEMV_H

#include <immintrin.h>
#include <cstddef>
#include <cstdint>

namespace weft {

// Native AVX-512 In-Register INT4 Activation / Ternary Weight GEMV Kernel
// Activations: packed 4-bit unsigned [0..15], centered to [-8..7]
// Weights: ternary {-1, 0, +1} in int8
// Computes single-pass vector dot product with zero memory allocation.
inline float fused_int4_ternary_gemv_avx512(
    const uint8_t* __restrict__ act_int4,      // Packed 2 nibbles per byte, size: dim / 2
    const int8_t* __restrict__ weights_ternary,// {-1, 0, +1}, size: dim
    size_t dim,
    float act_scale = 0.125f,
    float weight_scale = 0.5f)
{
    __m512 vsum = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        // Load 8 bytes (16 nibbles)
        uint64_t bytes8 = *reinterpret_cast<const uint64_t*>(act_int4 + (i / 2));

        // Unpack into 16 int32s in temporary array
        int32_t unpacked_act[16];
        for (int b = 0; b < 8; ++b) {
            uint8_t byte = static_cast<uint8_t>((bytes8 >> (b * 8)) & 0xFF);
            unpacked_act[b * 2]     = static_cast<int32_t>(byte & 0x0F) - 8;
            unpacked_act[b * 2 + 1] = static_cast<int32_t>((byte >> 4) & 0x0F) - 8;
        }

        __m512i vi_act = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(unpacked_act));
        __m512 v_act = _mm512_cvtepi32_ps(vi_act);

        // Load 16 int8 ternary weights
        __m128i vi8_w = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights_ternary + i));
        __m512i vi32_w = _mm512_cvtepi8_epi32(vi8_w);
        __m512 v_w = _mm512_cvtepi32_ps(vi32_w);

        vsum = _mm512_fmadd_ps(v_act, v_w, vsum);
    }

    float dot = _mm512_reduce_add_ps(vsum);
    for (; i < dim; ++i) {
        uint8_t byte = act_int4[i / 2];
        int8_t nibble = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        dot += static_cast<float>(nibble - 8) * static_cast<float>(weights_ternary[i]);
    }

    return dot * act_scale * weight_scale;
}

} // namespace weft

#endif // WEFT_FUSED_INT4_TERNARY_GEMV_H
