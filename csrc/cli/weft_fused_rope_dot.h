#pragma once
// weft_fused_rope_dot.h
// AVX-512 Fused Multi-Head Rotary Dot Product (Item 5)
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Fuses RoPE dequantized phase application and query-key dot product:
// Q_rot[i] = Q[i] - scale * Q_corr[i]
// K_rot[i] = K[i] - scale * K_corr[i]
// dot_acc += Q_rot[i] * K_rot[i]
inline float avx512_fused_rope_dot_product(
    const float* __restrict__ Q,
    const float* __restrict__ Q_corr,
    const float* __restrict__ K,
    const float* __restrict__ K_corr,
    size_t head_dim,
    float scale
) {
    __m512 v_scale = _mm512_set1_ps(scale);
    __m512 v_acc = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= head_dim; i += 16) {
        __m512 vq = _mm512_loadu_ps(Q + i);
        __m512 vqc = _mm512_loadu_ps(Q_corr + i);
        __m512 vk = _mm512_loadu_ps(K + i);
        __m512 vkc = _mm512_loadu_ps(K_corr + i);

        __m512 vq_rot = _mm512_fnmadd_ps(vqc, v_scale, vq);
        __m512 vk_rot = _mm512_fnmadd_ps(vkc, v_scale, vk);

        v_acc = _mm512_fmadd_ps(vq_rot, vk_rot, v_acc);
    }

    float dot = _mm512_reduce_add_ps(v_acc);
    for (; i < head_dim; ++i) {
        float q_rot = Q[i] - scale * Q_corr[i];
        float k_rot = K[i] - scale * K_corr[i];
        dot += q_rot * k_rot;
    }

    return dot;
}

} // namespace weft
