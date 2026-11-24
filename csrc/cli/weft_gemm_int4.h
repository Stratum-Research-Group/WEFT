#pragma once
// weft_gemm_int4.h
// AVX-512 Vectorized 4-Bit Packed Integer Matrix Multiplication Engine (GemmInt4) (Item 4)
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace weft {

// Computes matrix-vector multiplication y = A * x + bias
// where matrix A (M x N) is packed at 4 bits/weight (2 weights per byte)
// low nibble: weights[2*k], high nibble: weights[2*k + 1] (unsigned 0..15, zero_point = 8)
inline void avx512_gemv_int4(
    const uint8_t* __restrict__ packed_A,
    const float* __restrict__ x,
    const float* __restrict__ scales,
    const float* __restrict__ bias,
    float* __restrict__ y,
    size_t M,
    size_t N
) {
    size_t row_bytes = N / 2;

    for (size_t m = 0; m < M; ++m) {
        const uint8_t* row_ptr = packed_A + m * row_bytes;
        float row_scale = scales[m];
        __m512 v_scale = _mm512_set1_ps(row_scale);
        __m512 v_acc = _mm512_setzero_ps();

        size_t n = 0;
        size_t b_idx = 0;
        for (; n + 16 <= N; n += 16) {
            __m512 vx = _mm512_loadu_ps(x + n);

            alignas(64) float weights[16];
            for (int b = 0; b < 8; ++b) {
                uint8_t byte_val = row_ptr[b_idx++];
                int w_low = (byte_val & 0x0F) - 8;
                int w_high = ((byte_val >> 4) & 0x0F) - 8;
                weights[b * 2] = static_cast<float>(w_low);
                weights[b * 2 + 1] = static_cast<float>(w_high);
            }

            __m512 vw = _mm512_load_ps(weights);
            v_acc = _mm512_fmadd_ps(vx, vw, v_acc);
        }

        float dot = _mm512_reduce_add_ps(v_acc) * row_scale;
        if (bias) {
            dot += bias[m];
        }
        y[m] = dot;
    }
}

} // namespace weft
