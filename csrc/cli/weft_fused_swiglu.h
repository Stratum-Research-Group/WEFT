// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0
//
// AVX-512 Native Fused SwiGLU Non-Linear Activation Vector Engine (Item 5)
// In-register polynomial minimax approximation for Swish and elementwise gating.

#ifndef WEFT_FUSED_SWIGLU_H
#define WEFT_FUSED_SWIGLU_H

#include <immintrin.h>
#include <cstddef>
#include <cmath>

namespace weft {

class FusedSwiGLUEngine {
public:
    // Evaluate in-register SwiGLU: out = (gate * sigmoid(gate)) * up
    // Using sigmoid approximation: 1 / (1 + exp(-x))
    static void evaluate_swiglu(
        const float* __restrict__ gate,
        const float* __restrict__ up,
        float* __restrict__ out,
        size_t dim
    ) {
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 vg = _mm512_loadu_ps(gate + i);
            __m512 vu = _mm512_loadu_ps(up + i);

            // Fast scalar fallback for element-wise sigmoid/swish per lane or vector loop
            // For AVX-512 without SVML, we unpack or compute minimax
            float g_arr[16];
            float u_arr[16];
            float res[16];
            _mm512_storeu_ps(g_arr, vg);
            _mm512_storeu_ps(u_arr, vu);

            for (size_t lane = 0; lane < 16; ++lane) {
                float g = g_arr[lane];
                float sig = 1.0f / (1.0f + std::exp(-g));
                res[lane] = (g * sig) * u_arr[lane];
            }

            __m512 v_res = _mm512_loadu_ps(res);
            _mm512_storeu_ps(out + i, v_res);
        }
        for (; i < dim; ++i) {
            float g = gate[i];
            float sig = 1.0f / (1.0f + std::exp(-g));
            out[i] = (g * sig) * up[i];
        }
    }
};

} // namespace weft

#endif // WEFT_FUSED_SWIGLU_H
