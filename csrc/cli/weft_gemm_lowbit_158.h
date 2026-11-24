#pragma once
// weft_gemm_lowbit_158.h
// AVX-512 Vectorized Asymmetric 1.58-bit Ternary Matrix-Vector Engine (GemmLowBit) (Item 4)
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace weft {

// Computes full matrix-vector multiplication y = A * x + bias
// where matrix A (M x N) contains 2-bit packed ternary weights {-1, 0, +1}
// N must be multiple of 16.
inline void avx512_gemv_158_bit(
    const uint8_t* __restrict__ packed_matrix_A,
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ y,
    size_t M,
    size_t N
) {
    size_t row_bytes = N / 4;

    for (size_t m = 0; m < M; ++m) {
        const uint8_t* row_ptr = packed_matrix_A + m * row_bytes;
        __m512 v_acc = _mm512_setzero_ps();

        size_t n = 0;
        size_t b_idx = 0;
        for (; n + 16 <= N; n += 16) {
            __m512 vx = _mm512_loadu_ps(x + n);

            alignas(64) float weights[16];
            for (int b = 0; b < 4; ++b) {
                uint8_t byte_val = row_ptr[b_idx++];
                for (int shift = 0; shift < 8; shift += 2) {
                    uint8_t code = (byte_val >> shift) & 0x03;
                    float w = 0.0f;
                    if (code == 1) w = 1.0f;
                    else if (code == 2) w = -1.0f;
                    weights[b * 4 + (shift / 2)] = w;
                }
            }

            __m512 vw = _mm512_load_ps(weights);
            v_acc = _mm512_fmadd_ps(vx, vw, v_acc);
        }

        float dot = _mm512_reduce_add_ps(v_acc);
        if (bias) {
            dot += bias[m];
        }
        y[m] = dot;
    }
}

} // namespace weft
