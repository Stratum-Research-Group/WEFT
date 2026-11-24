#pragma once
// weft_motivic_milnor_rope.h
// Milestone Paper 248: Native AVX-512 Motivic Milnor Rotary Positional Embeddings
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized 2D cyclotomic RoPE rotation using AVX-512:
// q0_new = q0 * cos_theta - q1 * sin_theta
// q1_new = q0 * sin_theta + q1 * cos_theta
inline void avx512_milnor_rope_rotate_2d(
    const float* __restrict__ in_q0,
    const float* __restrict__ in_q1,
    float* __restrict__ out_q0,
    float* __restrict__ out_q1,
    size_t count_pairs,
    float cos_theta,
    float sin_theta
) {
    __m512 v_cos = _mm512_set1_ps(cos_theta);
    __m512 v_sin = _mm512_set1_ps(sin_theta);

    size_t i = 0;
    for (; i + 16 <= count_pairs; i += 16) {
        __m512 v0 = _mm512_loadu_ps(in_q0 + i);
        __m512 v1 = _mm512_loadu_ps(in_q1 + i);

        // q0_new = v0 * cos - v1 * sin
        __m512 v0_new = _mm512_fnmadd_ps(v1, v_sin, _mm512_mul_ps(v0, v_cos));
        // q1_new = v0 * sin + v1 * cos
        __m512 v1_new = _mm512_fmadd_ps(v0, v_sin, _mm512_mul_ps(v1, v_cos));

        _mm512_storeu_ps(out_q0 + i, v0_new);
        _mm512_storeu_ps(out_q1 + i, v1_new);
    }

    for (; i < count_pairs; ++i) {
        float q0 = in_q0[i];
        float q1 = in_q1[i];
        out_q0[i] = q0 * cos_theta - q1 * sin_theta;
        out_q1[i] = q0 * sin_theta + q1 * cos_theta;
    }
}

} // namespace weft
