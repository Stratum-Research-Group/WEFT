#ifndef WEFT_DERIVED_TANNAKIAN_CATEGORIES_H
#define WEFT_DERIVED_TANNAKIAN_CATEGORIES_H

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace weft {

// AVX-512 Asymmetric Derived Tannakian Multi-Modal Fusion Kernel
// Projects continuous visual patch features to align with discrete sub-bit text tokens.
inline void tannakian_multimodal_fusion_avx512(
    const float* __restrict__ visual_patches,  // Shape: [num_patches, dim]
    const int8_t* __restrict__ text_ternary,   // Discrete text tokens {-1, 0, +1}, shape: [dim]
    float* __restrict__ fused_out,             // Output cross-modal attention scores, shape: [num_patches]
    size_t num_patches,
    size_t dim,
    float scale = 0.125f)
{
    __m512 vscale = _mm512_set1_ps(scale);

    for (size_t p = 0; p < num_patches; ++p) {
        const float* patch_ptr = visual_patches + p * dim;
        __m512 vsum = _mm512_setzero_ps();

        size_t d = 0;
        for (; d + 16 <= dim; d += 16) {
            __m512 v_vis = _mm512_loadu_ps(patch_ptr + d);

            // Load 16 ternary int8 weights and convert to float
            __m128i vi8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(text_ternary + d));
            __m512i vi32 = _mm512_cvtepi8_epi32(vi8);
            __m512 v_txt = _mm512_cvtepi32_ps(vi32);

            vsum = _mm512_fmadd_ps(v_vis, v_txt, vsum);
        }

        float total = _mm512_reduce_add_ps(vsum);
        for (; d < dim; ++d) {
            total += patch_ptr[d] * static_cast<float>(text_ternary[d]);
        }

        fused_out[p] = total * scale;
    }
}

} // namespace weft

#endif // WEFT_DERIVED_TANNAKIAN_CATEGORIES_H
