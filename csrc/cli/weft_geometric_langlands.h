#ifndef WEFT_GEOMETRIC_LANGLANDS_H
#define WEFT_GEOMETRIC_LANGLANDS_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct HeckeOperatorKernel {
    int32_t shift;
    float weight;
};

// Computes discrete Hecke convolution: (H * W)[i] = sum_r weight_r * W[(i + shift_r) % dim]
inline void apply_hecke_operator_simd(const float* weights, size_t dim,
                                      const HeckeOperatorKernel* kernels, size_t num_kernels,
                                      float* output) {
    for (size_t i = 0; i < dim; ++i) {
        float sum = 0.0f;
        for (size_t k = 0; k < num_kernels; ++k) {
            size_t idx = (i + static_cast<size_t>(kernels[k].shift + dim)) % dim;
            sum += kernels[k].weight * weights[idx];
        }
        output[i] = sum;
    }
}

// In-register projection onto Hecke eigensheaf with eigenvalue lambda:
// W_proj = 0.5 * (W + (H * W) / lambda)
inline void project_hecke_eigensheaf_simd(float* weights, const float* hecke_applied, size_t dim, float lambda) {
    if (std::abs(lambda) < 1e-6f) lambda = 1.0f;
    float inv_lambda = 1.0f / lambda;
    size_t i = 0;
#if defined(__AVX2__)
    __m256 v_inv_lam = _mm256_set1_ps(inv_lambda);
    __m256 v_half = _mm256_set1_ps(0.5f);
    for (; i + 8 <= dim; i += 8) {
        __m256 v_w = _mm256_loadu_ps(&weights[i]);
        __m256 v_h = _mm256_loadu_ps(&hecke_applied[i]);
        __m256 v_h_scaled = _mm256_mul_ps(v_h, v_inv_lam);
        __m256 v_proj = _mm256_mul_ps(_mm256_add_ps(v_w, v_h_scaled), v_half);
        _mm256_storeu_ps(&weights[i], v_proj);
    }
#endif
    for (; i < dim; ++i) {
        weights[i] = 0.5f * (weights[i] + hecke_applied[i] * inv_lambda);
    }
}

} // namespace weft

#endif // WEFT_GEOMETRIC_LANGLANDS_H
