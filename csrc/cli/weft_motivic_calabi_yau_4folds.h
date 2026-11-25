#ifndef WEFT_MOTIVIC_CALABI_YAU_4FOLDS_H
#define WEFT_MOTIVIC_CALABI_YAU_4FOLDS_H

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace weft {

// AVX-512 Asymmetric Motivic Calabi-Yau 4-Fold Mixed-Precision Attention Kernel
// Evaluates fused dot-product between 4-bit INT4 high-curvature channels
// and 1.58-bit ternary background channels.
inline float cy4_mixed_precision_dot_avx512(
    const float* __restrict__ query,
    const uint8_t* __restrict__ key_int4,      // Packed 2 nibbles per byte, size: dim_4bit / 2
    const int8_t* __restrict__ key_ternary,    // {-1, 0, +1}, size: dim_ternary
    size_t dim_4bit,
    size_t dim_ternary,
    float scale_int4 = 0.125f,
    float scale_ternary = 0.666667f)
{
    __m512 vsum4 = _mm512_setzero_ps();
    __m512 vsum_ternary = _mm512_setzero_ps();

    // 1. Process 4-bit channels (dim_4bit elements)
    size_t i = 0;
    for (; i + 16 <= dim_4bit; i += 16) {
        __m512 vq = _mm512_loadu_ps(query + i);

        // Load 8 bytes (16 nibbles)
        uint64_t bytes8 = *reinterpret_cast<const uint64_t*>(key_int4 + (i / 2));
        
        // Unpack into 16 int32s in temporary array
        int32_t unpacked[16];
        for (int b = 0; b < 8; ++b) {
            uint8_t byte = static_cast<uint8_t>((bytes8 >> (b * 8)) & 0xFF);
            unpacked[b * 2]     = static_cast<int32_t>(byte & 0x0F) - 8;
            unpacked[b * 2 + 1] = static_cast<int32_t>((byte >> 4) & 0x0F) - 8;
        }

        __m512i vi4 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(unpacked));
        __m512 vk = _mm512_cvtepi32_ps(vi4);

        vsum4 = _mm512_fmadd_ps(vq, vk, vsum4);
    }

    float dot4 = _mm512_reduce_add_ps(vsum4) * scale_int4;
    for (; i < dim_4bit; ++i) {
        uint8_t byte = key_int4[i / 2];
        int8_t nibble = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        dot4 += query[i] * static_cast<float>(nibble - 8) * scale_int4;
    }

    // 2. Process ternary channels (dim_ternary elements)
    const float* query_ternary = query + dim_4bit;
    size_t j = 0;
    for (; j + 16 <= dim_ternary; j += 16) {
        __m512 vq = _mm512_loadu_ps(query_ternary + j);

        __m128i vi8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(key_ternary + j));
        __m512i vi32 = _mm512_cvtepi8_epi32(vi8);
        __m512 vk = _mm512_cvtepi32_ps(vi32);

        vsum_ternary = _mm512_fmadd_ps(vq, vk, vsum_ternary);
    }

    float dot_ternary = _mm512_reduce_add_ps(vsum_ternary) * scale_ternary;
    for (; j < dim_ternary; ++j) {
        dot_ternary += query_ternary[j] * static_cast<float>(key_ternary[j]) * scale_ternary;
    }

    return dot4 + dot_ternary;
}

} // namespace weft

#endif // WEFT_MOTIVIC_CALABI_YAU_4FOLDS_H
