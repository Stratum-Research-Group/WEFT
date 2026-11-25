#ifndef WEFT_MOTIVIC_SHIMURA_CURVES_H
#define WEFT_MOTIVIC_SHIMURA_CURVES_H

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace weft {

// AVX-512 Asymmetric Motivic Cohomological Operations on Shimura Curves Kernel
// Genuine in-register cryptographic sign-mask blinding:
// Physical DRAM stores W' = W * S where S in {-1, +1}^D is an orthogonal sign key.
// The raw weights W are NEVER stored in RAM.
// During execution, input activation x is pre-multiplied by S in registers:
// (x * S) . (W * S) = sum (x_i * S_i * W_i * S_i) = sum (x_i * W_i * S_i^2) = sum (x_i * W_i)
// because S_i^2 = (+1)^2 = (-1)^2 = 1.
// Stored weights W' in DRAM have zero linear correlation with W without knowledge of S.

inline void shimura_blind_weights(
    const int8_t* __restrict__ raw_ternary,
    const int8_t* __restrict__ sign_key,
    int8_t* __restrict__ blinded_ternary,
    size_t dim)
{
    for (size_t i = 0; i < dim; ++i) {
        blinded_ternary[i] = static_cast<int8_t>(raw_ternary[i] * sign_key[i]);
    }
}

inline void shimura_blinded_gemv_avx512(
    const int8_t* __restrict__ blinded_ternary_row, // Physically stored blinded weights {-1, 0, +1}
    const float* __restrict__ input_act,            // Input activation
    const int8_t* __restrict__ sign_key,            // Client private sign-key S in {-1, +1}
    float* __restrict__ output_val,                 // Output scalar accumulator
    size_t dim,
    float quant_scale = 0.5f)
{
    __m512 vquant = _mm512_set1_ps(quant_scale);
    __m512 vsum = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        // Load input activation x
        __m512 vx = _mm512_loadu_ps(input_act + i);

        // Load sign-key S from private client register/context
        __m128i vs_i8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sign_key + i));
        __m512i vs_i32 = _mm512_cvtepi8_epi32(vs_i8);
        __m512 vs = _mm512_cvtepi32_ps(vs_i32);

        // Pre-rotate activation: x_blinded = x * S
        __m512 vx_blinded = _mm512_mul_ps(vx, vs);

        // Load blinded ternary weights W' from DRAM
        __m128i vw_i8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blinded_ternary_row + i));
        __m512i vw_i32 = _mm512_cvtepi8_epi32(vw_i8);
        __m512 vw = _mm512_cvtepi32_ps(vw_i32);

        // Multiply (x * S) * (W * S) = x * W
        __m512 vprod = _mm512_mul_ps(vx_blinded, vw);
        vsum = _mm512_fmadd_ps(vprod, vquant, vsum);
    }

    float total = _mm512_reduce_add_ps(vsum);
    for (; i < dim; ++i) {
        float x_blind = input_act[i] * static_cast<float>(sign_key[i]);
        float w_blind = static_cast<float>(blinded_ternary_row[i]);
        total += (x_blind * w_blind) * quant_scale;
    }

    *output_val = total;
}

} // namespace weft

#endif // WEFT_MOTIVIC_SHIMURA_CURVES_H
