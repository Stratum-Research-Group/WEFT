#ifndef WEFT_MOTIVIC_QUIVER_GRASSMANNIANS_H
#define WEFT_MOTIVIC_QUIVER_GRASSMANNIANS_H

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <vector>

namespace weft {

// AVX-512 Asymmetric Motivic Quiver Grassmannians Dynamic Adapter Composer
// Fuses low-rank adapter projection onto output hidden state: y += lambda * A * (B * x)
inline void quiver_adapter_compose_avx512(
    const float* __restrict__ lora_A,  // Shape: [dim, rank]
    const float* __restrict__ lora_h,  // Intermediate projection: h = B * x, shape: [rank]
    float* __restrict__ output_y,      // Output hidden vector, shape: [dim]
    size_t dim,
    size_t rank,
    float lambda_scale = 1.0f)
{
    __m512 vlambda = _mm512_set1_ps(lambda_scale);

    for (size_t r = 0; r < rank; ++r) {
        float h_val = lora_h[r] * lambda_scale;
        __m512 vh = _mm512_set1_ps(h_val);
        const float* A_col = lora_A + r * dim;

        size_t d = 0;
        for (; d + 16 <= dim; d += 16) {
            __m512 vy = _mm512_loadu_ps(output_y + d);
            __m512 va = _mm512_loadu_ps(A_col + d);
            vy = _mm512_fmadd_ps(va, vh, vy);
            _mm512_storeu_ps(output_y + d, vy);
        }

        for (; d < dim; ++d) {
            output_y[d] += A_col[d] * h_val;
        }
    }
}

} // namespace weft

#endif // WEFT_MOTIVIC_QUIVER_GRASSMANNIANS_H
