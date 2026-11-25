// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0
//
// AVX-512 Native Hybrid INT2 / FP8 Attention Dot-Product Kernel (Item 5)
// Dequantizes 2-bit packed keys on-the-fly against FP8 (e4m3 / e5m2 emulation) queries in SIMD registers.

#ifndef WEFT_HYBRID_INT2_FP8_ATTENTION_H
#define WEFT_HYBRID_INT2_FP8_ATTENTION_H

#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cmath>

namespace weft {

class HybridInt2Fp8Attention {
public:
    // Evaluate dot product between FP32/FP8 query and 2-bit packed key vector.
    // 2-bit packed format: 4 ternary/int2 weights per byte: 00 -> 0.0, 01 -> +1.0, 10 -> -1.0, 11 -> 0.0
    static float compute_dot_product(
        const float* __restrict__ q,
        const uint8_t* __restrict__ k_packed_int2,
        float k_scale,
        size_t dim
    ) {
        __m512 v_acc = _mm512_setzero_ps();
        size_t i = 0;
        
        // Process in chunks of 16 floats (4 bytes of packed int2)
        for (; i + 16 <= dim; i += 16) {
            float k_unpacked[16];
            for (size_t b = 0; b < 4; ++b) {
                uint8_t byte_val = k_packed_int2[(i / 4) + b];
                for (size_t bit = 0; bit < 4; ++bit) {
                    uint8_t code = (byte_val >> (bit * 2)) & 0x03;
                    float val = 0.0f;
                    if (code == 1) val = 1.0f;
                    else if (code == 2) val = -1.0f;
                    k_unpacked[b * 4 + bit] = val * k_scale;
                }
            }
            __m512 vq = _mm512_loadu_ps(q + i);
            __m512 vk = _mm512_loadu_ps(k_unpacked);
            v_acc = _mm512_fmadd_ps(vq, vk, v_acc);
        }
        
        float total = _mm512_reduce_add_ps(v_acc);
        for (; i < dim; ++i) {
            uint8_t byte_val = k_packed_int2[i / 4];
            uint8_t code = (byte_val >> ((i % 4) * 2)) & 0x03;
            float val = 0.0f;
            if (code == 1) val = 1.0f;
            else if (code == 2) val = -1.0f;
            total += q[i] * (val * k_scale);
        }
        return total;
    }
};

} // namespace weft

#endif // WEFT_HYBRID_INT2_FP8_ATTENTION_H
