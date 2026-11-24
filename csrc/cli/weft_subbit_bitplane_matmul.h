#pragma once
// weft_subbit_bitplane_matmul.h
// AVX-512 Vectorized Bit-Plane Permutation for Sub-Bit MatMul (Item 4)
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace weft {

// Performs in-register dot product between float32 activation vector
// and 2-bit ternary weights {-1, 0, +1} unpacked on-the-fly without heap allocation.
inline float avx512_subbit_dot_product(
    const float* __restrict__ activations,
    const uint8_t* __restrict__ packed_ternary,
    size_t count // multiple of 16
) {
    __m512 v_acc = _mm512_setzero_ps();
    __m512 v_one = _mm512_set1_ps(1.0f);
    __m512 v_neg_one = _mm512_set1_ps(-1.0f);

    size_t i = 0;
    size_t byte_idx = 0;

    for (; i + 16 <= count; i += 16) {
        __m512 v_act = _mm512_loadu_ps(activations + i);

        // unpack 16 ternary values from 4 bytes
        alignas(64) float weights[16];
        for (int b = 0; b < 4; ++b) {
            uint8_t byte_val = packed_ternary[byte_idx++];
            for (int shift = 0; shift < 8; shift += 2) {
                uint8_t code = (byte_val >> shift) & 0x03;
                float w = 0.0f;
                if (code == 1) w = 1.0f;
                else if (code == 2) w = -1.0f;
                weights[b * 4 + (shift / 2)] = w;
            }
        }

        __m512 v_w = _mm512_load_ps(weights);
        v_acc = _mm512_fmadd_ps(v_act, v_w, v_acc);
    }

    float total = _mm512_reduce_add_ps(v_acc);

    for (; i < count; ++i) {
        // scalar remainder if count not multiple of 16
        uint8_t byte_val = packed_ternary[byte_idx];
        int shift = (i % 4) * 2;
        uint8_t code = (byte_val >> shift) & 0x03;
        float w = 0.0f;
        if (code == 1) w = 1.0f;
        else if (code == 2) w = -1.0f;
        total += activations[i] * w;
        if ((i % 4) == 3) byte_idx++;
    }

    return total;
}

} // namespace weft
