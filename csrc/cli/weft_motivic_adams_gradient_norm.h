#pragma once
// weft_motivic_adams_gradient_norm.h
// Milestone Paper 245: Native AVX-512 Motivic Adams Gradient Normalization
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

struct AdamsNormConfig {
    float k_factor; // Adams degree scaling (e.g. 2.0f)
    float lambda;   // Normalization bounding factor (e.g. 0.05f)
};

// Vectorized algebraic gradient normalization using AVX-512:
// out[i] = (k * g[i]) / (1.0 + lambda * k * |g[i]|)
inline void avx512_motivic_adams_normalize_grad(
    const float* __restrict__ in_grad,
    float* __restrict__ out_grad,
    size_t count,
    const AdamsNormConfig& cfg
) {
    __m512 v_k = _mm512_set1_ps(cfg.k_factor);
    __m512 v_lam_k = _mm512_set1_ps(cfg.lambda * cfg.k_factor);
    __m512 v_one = _mm512_set1_ps(1.0f);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vg = _mm512_loadu_ps(in_grad + i);
        __m512 v_num = _mm512_mul_ps(vg, v_k);

        __m512 v_abs_g = _mm512_abs_ps(vg);
        __m512 v_denom = _mm512_fmadd_ps(v_lam_k, v_abs_g, v_one);

        __m512 v_out = _mm512_div_ps(v_num, v_denom);
        _mm512_storeu_ps(out_grad + i, v_out);
    }

    for (; i < count; ++i) {
        float g = in_grad[i];
        float num = cfg.k_factor * g;
        float denom = 1.0f + cfg.lambda * cfg.k_factor * std::fabs(g);
        out_grad[i] = num / denom;
    }
}

} // namespace weft
