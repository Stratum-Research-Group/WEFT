#pragma once
// weft_motivic_cobordism_norm.h
// Milestone Paper 239: Native AVX-512 Motivic Cobordism Layer Normalization
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

struct CobordismNormConfig {
    float alpha; // Disk contraction parameter (e.g. 0.05)
    float a1;    // Formal group second-order coefficient (e.g. 0.25)
};

// Vectorized continuous formal group normalization using AVX-512:
// u_i = x_i / (1 + alpha * |x_i|)
// log_F(u)_i = u_i - a1 * u_i * |u_i|
inline void avx512_motivic_cobordism_normalize(
    const float* __restrict__ in_activations,
    float* __restrict__ out_normalized,
    size_t count,
    const CobordismNormConfig& cfg
) {
    __m512 v_alpha = _mm512_set1_ps(cfg.alpha);
    __m512 v_one = _mm512_set1_ps(1.0f);
    __m512 v_a1 = _mm512_set1_ps(cfg.a1);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vx = _mm512_loadu_ps(in_activations + i);
        __m512 v_abs_x = _mm512_abs_ps(vx);

        // denom = 1 + alpha * |x|
        __m512 v_denom = _mm512_fmadd_ps(v_alpha, v_abs_x, v_one);
        __m512 vu = _mm512_div_ps(vx, v_denom);
        __m512 v_abs_u = _mm512_abs_ps(vu);

        // out = vu - a1 * vu * |vu|
        __m512 v_quad = _mm512_mul_ps(vu, v_abs_u);
        __m512 v_out = _mm512_fnmadd_ps(v_a1, v_quad, vu);

        _mm512_storeu_ps(out_normalized + i, v_out);
    }

    for (; i < count; ++i) {
        float x = in_activations[i];
        float abs_x = std::fabs(x);
        float u = x / (1.0f + cfg.alpha * abs_x);
        out_normalized[i] = u - cfg.a1 * u * std::fabs(u);
    }
}

} // namespace weft
