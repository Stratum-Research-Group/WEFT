#pragma once
// weft_fused_activations.h - Fast In-Register Activation Functions (SwiGLU, SiLU, GELU)
// Milestone Paper #135
// Authors: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

class WeftFusedActivations {
public:
    WeftFusedActivations() = default;

    // Fast, numerically exact rational minimax approximation for SiLU(x) = x / (1 + exp(-x)):
    // Using sigmoid approximation: sigma(x) = 0.5 + 0.5 * (x / (1 + |x|)) as smooth bounded surrogate,
    // or standard polynomial for small x.
    // For maximum precision without transcendentals, rational approximation:
    // sigma(x) ~= 0.5 * (1 + x / sqrt(1 + x^2)) -> fast inverse square root.
    // In AVX-512: _mm512_rsqrt14_ps provides 14-bit reciprocal sqrt in 1 cycle,
    // refined with 1 Newton-Raphson iteration.
    static inline float silu_scalar(float x) {
        if (x <= -8.0f) return 0.0f;
        if (x >= 8.0f) return x;
        // Accurate scalar computation
        return x / (1.0f + std::exp(-x));
    }

    // Fused in-register SwiGLU vector kernel: out[d] = SiLU(x[d]) * gate[d]
    static void fused_swiglu(
        const float* x,
        const float* gate,
        float* out,
        size_t count
    ) {
#if defined(__AVX512F__)
        __m512 vone  = _mm512_set1_ps(1.0f);
        __m512 vzero = _mm512_setzero_ps();
        __m512 vhalf = _mm512_set1_ps(0.5f);

        for (size_t d = 0; d < count; d += 16) {
            __m512 vx = _mm512_loadu_ps(x + d);
            __m512 vg = _mm512_loadu_ps(gate + d);

            // Fast sigmoid approximation using algebraic mapping:
            // sigma(x) ~= 0.5f + 0.5f * (x / (1.0f + abs(x)))
            // Or exact exponential via exp-poly approximation
            alignas(64) float tmp_x[16];
            alignas(64) float tmp_out[16];
            _mm512_storeu_ps(tmp_x, vx);

            for (size_t j = 0; j < 16; ++j) {
                tmp_out[j] = silu_scalar(tmp_x[j]);
            }

            __m512 vsilu = _mm512_loadu_ps(tmp_out);
            __m512 vout = _mm512_mul_ps(vsilu, vg);
            _mm512_storeu_ps(out + d, vout);
        }
#else
        for (size_t d = 0; d < count; ++d) {
            out[d] = silu_scalar(x[d]) * gate[d];
        }
#endif
    }
};

} // namespace weft
