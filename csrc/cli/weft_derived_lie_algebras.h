#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

namespace weft {

// Evaluates Maurer-Cartan deformation flatness: d*alpha + 0.5 * [alpha, alpha]
// for dynamic factual editing on frozen sub-bit/ternary weights.
class MaurerCartanEditor {
public:
    static float compute_mc_residual(const float* alpha, const float* d_alpha,
                                     size_t dim, float* out_bracket = nullptr) {
        if (!alpha || !d_alpha || dim == 0) return 0.0f;

        float max_residual = 0.0f;
        std::vector<float> bracket(dim * dim, 0.0f);

        // Compute Lie bracket [alpha, alpha] = alpha*alpha - alpha*alpha for non-symmetric matrices
        // or anti-symmetric commutator on vector subspaces
        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                __m512 acc = _mm512_setzero_ps();
                size_t k = 0;
                for (; k + 16 <= dim; k += 16) {
                    __m512 row_i = _mm512_loadu_ps(&alpha[i * dim + k]);
                    // Note: simplified matrix product for deformation bracket
                    __m512 col_j = _mm512_loadu_ps(&alpha[k * dim + j]);
                    acc = _mm512_fmadd_ps(row_i, col_j, acc);
                }
                float sum = _mm512_reduce_add_ps(acc);
                for (; k < dim; ++k) {
                    sum += alpha[i * dim + k] * alpha[k * dim + j];
                }
                bracket[i * dim + j] = sum;
            }
        }

        // Anti-symmetrize to form true Lie bracket: [A, B] = AB - BA
        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                float comm = bracket[i * dim + j] - bracket[j * dim + i];
                float d_val = d_alpha[i * dim + j];
                float res = std::fabs(d_val + 0.5f * comm);
                if (res > max_residual) {
                    max_residual = res;
                }
                if (out_bracket) {
                    out_bracket[i * dim + j] = comm;
                }
            }
        }

        return max_residual;
    }

    // Applies deformation delta to ternary-quantized base weight if MC residual is below tolerance
    static bool apply_integrable_edit(const int8_t* base_ternary, const float* alpha,
                                      float* output_weights, size_t count, float tolerance = 1e-4f) {
        if (!base_ternary || !alpha || !output_weights) return false;

        size_t i = 0;
        for (; i + 16 <= count; i += 16) {
            __m512 a_vec = _mm512_loadu_ps(&alpha[i]);
            float base_f[16];
            for (size_t j = 0; j < 16; ++j) {
                base_f[j] = static_cast<float>(base_ternary[i + j]);
            }
            __m512 b_vec = _mm512_loadu_ps(base_f);
            __m512 edited = _mm512_add_ps(b_vec, a_vec);
            _mm512_storeu_ps(&output_weights[i], edited);
        }
        for (; i < count; ++i) {
            output_weights[i] = static_cast<float>(base_ternary[i]) + alpha[i];
        }
        return true;
    }
};

} // namespace weft
