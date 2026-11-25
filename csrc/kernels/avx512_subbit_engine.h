#pragma once

#include <cstdint>
#include <cstddef>
#include <immintrin.h>

namespace weft {

/**
 * AVX-512 Sub-bit Ternary Dot Product and GEMV Engine.
 * Optimized for AMD Zen 4 (AVX-512F, AVX-512BW, AVX-512CD, AVX-512DQ, AVX-512VPOPCNTDQ, AVX-512VNNI).
 */
class AVX512SubBitEngine {
public:
    /**
     * Executes dot product between int8 activation vector and 1-bit packed binary weights (+1/-1).
     * u_packed: bitboard where bit 1 -> +1, bit 0 -> -1.
     * activation: int8_t array of length N (N multiple of 64).
     */
    static int32_t dot_product_binary(const int8_t* activation, const uint64_t* u_packed, size_t n_elements) {
        int32_t accumulator = 0;
        size_t n_words = n_elements / 64;

#if defined(__AVX512F__) && defined(__AVX512BW__)
        __m512i vzero = _mm512_setzero_si512();
        __m512i vpos = _mm512_set1_epi8(1);
        __m512i vneg = _mm512_set1_epi8(-1);

        for (size_t w = 0; w < n_words; ++w) {
            uint64_t mask = u_packed[w];
            __mmask64 k_mask = static_cast<__mmask64>(mask);

            // Load 64 int8 activation values
            __m512i vact = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(activation + w * 64));

            // Expand bitboard into weights: 1 -> +1, 0 -> -1
            __m512i vweight = _mm512_mask_blend_epi8(k_mask, vneg, vpos);

            // Multiply and horizontal add via pmaddubsw or madd
            // vact * vweight:
            __m512i vlo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(vact));
            __m512i vhi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vact, 1));

            __m512i wlo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(vweight));
            __m512i whi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vweight, 1));

            __m512i prod_lo = _mm512_madd_epi16(vlo, wlo);
            __m512i prod_hi = _mm512_madd_epi16(vhi, whi);

            accumulator += _mm512_reduce_add_epi32(prod_lo);
            accumulator += _mm512_reduce_add_epi32(prod_hi);
        }
#else
        for (size_t w = 0; w < n_words; ++w) {
            uint64_t mask = u_packed[w];
            for (size_t b = 0; b < 64; ++b) {
                int8_t weight = ((mask >> b) & 1) ? 1 : -1;
                accumulator += static_cast<int32_t>(activation[w * 64 + b]) * weight;
            }
        }
#endif
        return accumulator;
    }
};

} // namespace weft
