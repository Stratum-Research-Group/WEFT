#pragma once
// weft_tropical_toric_quant_grid.h
// Milestone Paper 241: Native AVX-512 Tropical Toric Non-Uniform Quantization Grids
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <immintrin.h>

namespace weft {

struct ToricQuantConfig {
    float delta_base; // Base grid step (e.g. 0.1f)
    float gamma;      // Quadratic outward expansion (e.g. 0.05f)
};

// Vectorized non-uniform dequantization using AVX-512:
// out[i] = sign(q[i] - 7.5) * delta_base * (|q[i] - 7.5| + gamma * (q[i] - 7.5)^2)
inline void avx512_toric_dequantize_grid(
    const float* __restrict__ q_levels,
    float* __restrict__ out_values,
    size_t count,
    const ToricQuantConfig& cfg
) {
    __m512 v_center = _mm512_set1_ps(7.5f);
    __m512 v_delta = _mm512_set1_ps(cfg.delta_base);
    __m512 v_gamma = _mm512_set1_ps(cfg.gamma);
    __m512 v_one = _mm512_set1_ps(1.0f);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vq = _mm512_loadu_ps(q_levels + i);
        __m512 v_diff = _mm512_sub_ps(vq, v_center);

        // Compute polynomial: v_diff * delta * (1.0 + gamma * |v_diff|)
        __m512 v_abs = _mm512_abs_ps(v_diff);
        __m512 v_factor = _mm512_fmadd_ps(v_gamma, v_abs, v_one);
        __m512 v_out = _mm512_mul_ps(_mm512_mul_ps(v_diff, v_delta), v_factor);

        _mm512_storeu_ps(out_values + i, v_out);
    }

    for (; i < count; ++i) {
        float diff = q_levels[i] - 7.5f;
        float factor = 1.0f + cfg.gamma * std::fabs(diff);
        out_values[i] = diff * cfg.delta_base * factor;
    }
}

} // namespace weft
