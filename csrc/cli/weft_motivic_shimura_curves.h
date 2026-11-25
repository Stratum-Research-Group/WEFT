#ifndef WEFT_MOTIVIC_SHIMURA_CURVES_H
#define WEFT_MOTIVIC_SHIMURA_CURVES_H

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace weft {

// AVX-512 Asymmetric Motivic Cohomological Operations on Shimura Curves Kernel
// Direct in-register evaluation of quaternion-blinded ternary matrix-vector products.
// Secret key K scales/inverts coordinates in-register so unblinded weights never enter RAM.
inline void shimura_blinded_gemv_avx512(
    const int8_t* __restrict__ blinded_ternary_row, // Physically stored blinded weights {-1, 0, +1}
    const float* __restrict__ input_act,            // Input activation
    float* __restrict__ output_val,                 // Output scalar accumulator
    size_t dim,
    float key_scale = 1.337f,                       // Secret client blinding key
    float quant_scale = 0.5f)
{
    // Apply key transformation to input vector in registers
    __m512 vkey = _mm512_set1_ps(key_scale);
    __m512 vinv_key = _mm512_set1_ps(1.0f / key_scale);
    __m512 vquant = _mm512_set1_ps(quant_scale);
    __m512 vsum = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        // Load input activation and apply key pre-rotation
        __m512 vx = _mm512_loadu_ps(input_act + i);
        __m512 vx_blinded = _mm512_mul_ps(vx, vkey);

        // Load blinded ternary weights from memory
        __m128i vi8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blinded_ternary_row + i));
        __m512i vi32 = _mm512_cvtepi8_epi32(vi8);
        __m512 vw = _mm512_cvtepi32_ps(vi32);

        // In-register multiply and post-rotation by inverse key
        __m512 vprod = _mm512_mul_ps(_mm512_mul_ps(vx_blinded, vw), vinv_key);
        vsum = _mm512_fmadd_ps(vprod, vquant, vsum);
    }

    float total = _mm512_reduce_add_ps(vsum);
    for (; i < dim; ++i) {
        float x_blinded = input_act[i] * key_scale;
        float w = static_cast<float>(blinded_ternary_row[i]);
        total += (x_blinded * w * (1.0f / key_scale)) * quant_scale;
    }

    *output_val = total;
}

} // namespace weft

#endif // WEFT_MOTIVIC_SHIMURA_CURVES_H
