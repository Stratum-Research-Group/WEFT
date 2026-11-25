#pragma once

#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <cstring>
#include <stdexcept>

namespace weft {

/**
 * High-performance Sub-Bit Matrix Transpose & Dequantization routines.
 * Designed for AVX2 / AVX-512 register architectures to stream packed ternary
 * (-1, 0, +1) weights into floating-point activation registers.
 */
class SubBitTranspose {
public:
    /**
     * Unpacks 2-bit packed ternary values (4 values per byte) and unpacks/transposes
     * into a block of 32-bit floats.
     * Mapping:
     *   00 ->  0.0f
     *   01 -> +1.0f
     *   10 -> -1.0f
     *   11 ->  0.0f (reserved/padding)
     */
    static void dequantize_ternary_block_avx2(
        const uint8_t* __restrict src_packed,
        float* __restrict dst_fp32,
        size_t num_bytes,
        float scale
    ) {
        size_t num_values = num_bytes * 4;
        size_t i = 0;

        #if defined(__AVX2__)
        __m256 v_scale = _mm256_set1_ps(scale);
        __m256 v_zero = _mm256_setzero_ps();
        __m256 v_one = _mm256_set1_ps(1.0f);
        __m256 v_neg_one = _mm256_set1_ps(-1.0f);

        // Process 8 bytes (32 ternary elements) per loop iteration
        for (; i + 8 <= num_bytes; i += 8) {
            // Scalar unpack into temporary buffer, then vectorize scale
            alignas(32) float temp[32];
            for (size_t b = 0; b < 8; ++b) {
                uint8_t byte_val = src_packed[i + b];
                for (int bit_idx = 0; bit_idx < 4; ++bit_idx) {
                    uint8_t code = (byte_val >> (bit_idx * 2)) & 0x03;
                    float val = 0.0f;
                    if (code == 1) val = 1.0f;
                    else if (code == 2) val = -1.0f;
                    temp[b * 4 + bit_idx] = val;
                }
            }

            // Apply scaling via AVX2 FMA/mul
            for (int k = 0; k < 32; k += 8) {
                __m256 v_raw = _mm256_loadu_ps(&temp[k]);
                __m256 v_res = _mm256_mul_ps(v_raw, v_scale);
                _mm256_storeu_ps(&dst_fp32[(i + 0) * 4 + k], v_res);
            }
        }
        #endif

        // Scalar fallback for tail elements
        for (; i < num_bytes; ++i) {
            uint8_t byte_val = src_packed[i];
            for (int bit_idx = 0; bit_idx < 4; ++bit_idx) {
                uint8_t code = (byte_val >> (bit_idx * 2)) & 0x03;
                float val = 0.0f;
                if (code == 1) val = scale;
                else if (code == 2) val = -scale;
                dst_fp32[i * 4 + bit_idx] = val;
            }
        }
    }

    /**
     * In-place or out-of-place 8x8 block matrix transpose for 32-bit floats using AVX2.
     */
    static void transpose_8x8_fp32_avx2(
        const float* __restrict src,
        float* __restrict dst,
        size_t lda,
        size_t ldb
    ) {
        #if defined(__AVX2__)
        __m256 row0 = _mm256_loadu_ps(&src[0 * lda]);
        __m256 row1 = _mm256_loadu_ps(&src[1 * lda]);
        __m256 row2 = _mm256_loadu_ps(&src[2 * lda]);
        __m256 row3 = _mm256_loadu_ps(&src[3 * lda]);
        __m256 row4 = _mm256_loadu_ps(&src[4 * lda]);
        __m256 row5 = _mm256_loadu_ps(&src[5 * lda]);
        __m256 row6 = _mm256_loadu_ps(&src[6 * lda]);
        __m256 row7 = _mm256_loadu_ps(&src[7 * lda]);

        __m256 t0 = _mm256_unpacklo_ps(row0, row1);
        __m256 t1 = _mm256_unpackhi_ps(row0, row1);
        __m256 t2 = _mm256_unpacklo_ps(row2, row3);
        __m256 t3 = _mm256_unpackhi_ps(row2, row3);
        __m256 t4 = _mm256_unpacklo_ps(row4, row5);
        __m256 t5 = _mm256_unpackhi_ps(row4, row5);
        __m256 t6 = _mm256_unpacklo_ps(row6, row7);
        __m256 t7 = _mm256_unpackhi_ps(row6, row7);

        __m256 tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
        __m256 tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

        _mm256_storeu_ps(&dst[0 * ldb], _mm256_permute2f128_ps(tt0, tt4, 0x20));
        _mm256_storeu_ps(&dst[1 * ldb], _mm256_permute2f128_ps(tt1, tt5, 0x20));
        _mm256_storeu_ps(&dst[2 * ldb], _mm256_permute2f128_ps(tt2, tt6, 0x20));
        _mm256_storeu_ps(&dst[3 * ldb], _mm256_permute2f128_ps(tt3, tt7, 0x20));
        _mm256_storeu_ps(&dst[4 * ldb], _mm256_permute2f128_ps(tt0, tt4, 0x31));
        _mm256_storeu_ps(&dst[5 * ldb], _mm256_permute2f128_ps(tt1, tt5, 0x31));
        _mm256_storeu_ps(&dst[6 * ldb], _mm256_permute2f128_ps(tt2, tt6, 0x31));
        _mm256_storeu_ps(&dst[7 * ldb], _mm256_permute2f128_ps(tt3, tt7, 0x31));
        #else
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                dst[c * ldb + r] = src[r * lda + c];
            }
        }
        #endif
    }
};

} // namespace weft
