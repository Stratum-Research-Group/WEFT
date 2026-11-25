#ifndef WEFT_FUSED_GELU_PADE_H
#define WEFT_FUSED_GELU_PADE_H

#include <immintrin.h>
#include <cstddef>
#include <cmath>
#include <algorithm>

namespace weft {

// AVX-512 Fused GeLU Activation with In-Register Rational Padé Approximation
// GeLU(x) = x * P(x) / Q(x)
// Eliminates transcendental functions (exp / erf / tanh) entirely using pure vector FMA.
// Approximates 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// via a [4/4] minimax rational Padé approximant on x.
inline void fused_gelu_pade_avx512(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t size)
{
    // Padé coefficients for standard normal CDF Phi(x):
    // Phi(x) ~ 1 / (1 + exp(-1.702 * x))
    // We approximate sigmoid(1.702 * x) via rational polynomial on [-4, 4]
    // P(y) = y * (p0 + p1 * y^2 + p2 * y^4)
    // Q(y) = q0 + q1 * y^2 + q2 * y^4
    // Alternatively, standard tanh rational approximation:
    // tanh(z) ~ z * (27 + z^2) / (27 + 9 * z^2)
    // Here z = 0.79788456f * (x + 0.044715f * x^3)

    const __m512 v_c0 = _mm512_set1_ps(0.7978845608f); // sqrt(2/pi)
    const __m512 v_c1 = _mm512_set1_ps(0.044715f);
    const __m512 v_half = _mm512_set1_ps(0.5f);
    const __m512 v_one = _mm512_set1_ps(1.0f);
    const __m512 v_27 = _mm512_set1_ps(27.0f);
    const __m512 v_9 = _mm512_set1_ps(9.0f);
    const __m512 v_neg_three = _mm512_set1_ps(-3.5f);
    const __m512 v_pos_three = _mm512_set1_ps(3.5f);

    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        __m512 vx = _mm512_loadu_ps(input + i);
        __m512 vx2 = _mm512_mul_ps(vx, vx);
        __m512 vx3 = _mm512_mul_ps(vx2, vx);

        // z = c0 * (x + c1 * x^3)
        __m512 vz = _mm512_mul_ps(v_c0, _mm512_fmadd_ps(v_c1, vx3, vx));

        // Clamp z to [-3.5, 3.5] for rational stability
        vz = _mm512_max_ps(v_neg_three, _mm512_min_ps(v_pos_three, vz));

        __m512 vz2 = _mm512_mul_ps(vz, vz);

        // Num = z * (27 + z^2)
        __m512 vnum = _mm512_mul_ps(vz, _mm512_add_ps(v_27, vz2));

        // Den = 27 + 9 * z^2
        __m512 vden = _mm512_fmadd_ps(v_9, vz2, v_27);

        // tanh_z ~ Num / Den
        __m512 vtanh = _mm512_div_ps(vnum, vden);

        // GeLU = 0.5 * x * (1.0 + tanh)
        __m512 vgelu = _mm512_mul_ps(_mm512_mul_ps(v_half, vx), _mm512_add_ps(v_one, vtanh));

        _mm512_storeu_ps(output + i, vgelu);
    }

    for (; i < size; ++i) {
        float x = input[i];
        float z = 0.7978845608f * (x + 0.044715f * x * x * x);
        z = std::clamp(z, -3.5f, 3.5f);
        float z2 = z * z;
        float tanh_z = (z * (27.0f + z2)) / (27.0f + 9.0f * z2);
        output[i] = 0.5f * x * (1.0f + tanh_z);
    }
}

} // namespace weft

#endif // WEFT_FUSED_GELU_PADE_H
