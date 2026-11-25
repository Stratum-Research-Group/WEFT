// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0
//
// Milestone Paper 280: Asymmetric Tropical Compactifications & Moduli of Metric Graphs
// for Dynamic Long-Context RoPE Scaling.
//
// AVX-512 SIMD kernel for scaled rotary position embedding.

#ifndef WEFT_TROPICAL_MODULI_ROPE_H
#define WEFT_TROPICAL_MODULI_ROPE_H

#include <immintrin.h>
#include <cstddef>
#include <cmath>

namespace weft {

class TropicalRoPEEngine {
public:
    // Apply rotary position embedding on vector x using precomputed cos/sin tables
    static void apply_rope(
        const float* __restrict__ x,
        const float* __restrict__ cos_tab,
        const float* __restrict__ sin_tab,
        float* __restrict__ x_out,
        size_t half_dim
    ) {
        size_t i = 0;
        for (; i + 16 <= half_dim; i += 16) {
            __m512 v_c = _mm512_loadu_ps(cos_tab + i);
            __m512 v_s = _mm512_loadu_ps(sin_tab + i);

            // x[2*i] and x[2*i + 1] interleaved
            // For simplicity and vectorization, assuming x has first half and second half separated
            __m512 vx0 = _mm512_loadu_ps(x + i);
            __m512 vx1 = _mm512_loadu_ps(x + half_dim + i);

            __m512 vy0 = _mm512_fmsub_ps(vx0, v_c, _mm512_mul_ps(vx1, v_s));
            __m512 vy1 = _mm512_fmadd_ps(vx0, v_s, _mm512_mul_ps(vx1, v_c));

            _mm512_storeu_ps(x_out + i, vy0);
            _mm512_storeu_ps(x_out + half_dim + i, vy1);
        }
        for (; i < half_dim; ++i) {
            float c = cos_tab[i];
            float s = sin_tab[i];
            float x0 = x[i];
            float x1 = x[half_dim + i];
            x_out[i] = x0 * c - x1 * s;
            x_out[half_dim + i] = x0 * s + x1 * c;
        }
    }
};

} // namespace weft

#endif // WEFT_TROPICAL_MODULI_ROPE_H
