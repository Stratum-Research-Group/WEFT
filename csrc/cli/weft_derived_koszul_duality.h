#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

namespace weft {

// Operadic quadratic tree composition evaluator for multi-modal activation fusion.
// Evaluates tree composition: (x o_1 y) projected onto Koszul-dual relations.
class OperadicKoszulComposer {
public:
    static void compose_binary_tree(const float* modality_a, const float* modality_b,
                                    float* fused_output, size_t dim, float alpha = 0.5f) {
        if (!modality_a || !modality_b || !fused_output || dim == 0) return;

        __m512 alpha_vec = _mm512_set1_ps(alpha);
        __m512 beta_vec = _mm512_set1_ps(1.0f - alpha);

        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 a = _mm512_loadu_ps(&modality_a[i]);
            __m512 b = _mm512_loadu_ps(&modality_b[i]);
            __m512 fused = _mm512_add_ps(_mm512_mul_ps(a, alpha_vec), _mm512_mul_ps(b, beta_vec));
            _mm512_storeu_ps(&fused_output[i], fused);
        }
        for (; i < dim; ++i) {
            fused_output[i] = modality_a[i] * alpha + modality_b[i] * (1.0f - alpha);
        }
    }

    // Projects fused multi-modal token vector against ternary weights W in {-1, 0, 1}
    static float evaluate_operad_norm(const float* fused_vector, const int8_t* ternary_kernel, size_t dim) {
        if (!fused_vector || !ternary_kernel || dim == 0) return 0.0f;

        __m512 acc = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 v = _mm512_loadu_ps(&fused_vector[i]);
            float k_f[16];
            for (size_t j = 0; j < 16; ++j) {
                k_f[j] = static_cast<float>(ternary_kernel[i + j]);
            }
            __m512 k = _mm512_loadu_ps(k_f);
            acc = _mm512_fmadd_ps(v, k, acc);
        }
        float total = _mm512_reduce_add_ps(acc);
        for (; i < dim; ++i) {
            total += fused_vector[i] * static_cast<float>(ternary_kernel[i]);
        }
        return total;
    }
};

} // namespace weft
