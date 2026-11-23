#pragma once
#include <immintrin.h>
#include <cmath>

// In-Register Vectorized Fast SiLU and SwiGLU Kernel for AVX-512
// Approximates sigmoid(x) = 1 / (1 + exp(-x)) using rational minimax polynomial or direct vector exp

static inline __m512 _mm512_silu_ps(__m512 x) {
    // Fast vector approximation of x * sigmoid(x)
    // Clamp range [-10.0, 10.0]
    __m512 c_min = _mm512_set1_ps(-10.0f);
    __m512 c_max = _mm512_set1_ps(10.0f);
    __m512 x_clamped = _mm512_min_ps(_mm512_max_ps(x, c_min), c_max);

    // Rational approximation: sigmoid(x) ≈ 0.5 + 0.5 * (x / (1 + |x|)) for fast in-register activation
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 half = _mm512_set1_ps(0.5f);
    __m512 abs_x = _mm512_abs_ps(x_clamped);
    __m512 denom = _mm512_add_ps(one, abs_x);
    __m512 ratio = _mm512_div_ps(x_clamped, denom);
    __m512 sig = _mm512_add_ps(half, _mm512_mul_ps(half, ratio));

    return _mm512_mul_ps(x, sig);
}

static inline void avx512_fused_swiglu(
    const float* gate,
    const float* up,
    float* out,
    int dim
) {
    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 vg = _mm512_loadu_ps(gate + i);
        __m512 vu = _mm512_loadu_ps(up + i);
        __m512 vsilu = _mm512_silu_ps(vg);
        __m512 vout = _mm512_mul_ps(vsilu, vu);
        _mm512_storeu_ps(out + i, vout);
    }
    for (; i < dim; ++i) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + std::exp(-g));
        out[i] = (g * sig) * up[i];
    }
}
