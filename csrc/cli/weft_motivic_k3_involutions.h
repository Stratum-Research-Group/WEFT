#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

namespace weft {

// Motivic K3 non-symplectic involution reflection masker for acoustic side-channel protection.
// Applies orthogonal Picard-Lefschetz reflections to randomize GEMV power signatures.
class K3SideChannelMasker {
public:
    static void apply_reflection_permutation(const float* input, float* masked_output,
                                             size_t dim, uint32_t seed = 0x5A5A5A5A) {
        if (!input || !masked_output || dim == 0) return;

        // Apply pseudo-random sign flipping based on involution eigenvalues (+1, -1)
        uint32_t state = seed;
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 in_vec = _mm512_loadu_ps(&input[i]);
            float signs[16];
            for (size_t j = 0; j < 16; ++j) {
                state = state * 1664525u + 1013904223u;
                signs[j] = (state & 1) ? 1.0f : -1.0f;
            }
            __m512 sign_vec = _mm512_loadu_ps(signs);
            __m512 out_vec = _mm512_mul_ps(in_vec, sign_vec);
            _mm512_storeu_ps(&masked_output[i], out_vec);
        }
        for (; i < dim; ++i) {
            state = state * 1664525u + 1013904223u;
            masked_output[i] = input[i] * ((state & 1) ? 1.0f : -1.0f);
        }
    }

    // Evaluates masked ternary dot product and restores correct invariant parity
    static float evaluate_masked_dot_product(const float* masked_input, const int8_t* ternary_weight,
                                             size_t dim, uint32_t seed = 0x5A5A5A5A) {
        if (!masked_input || !ternary_weight || dim == 0) return 0.0f;

        uint32_t state = seed;
        __m512 acc = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 m_in = _mm512_loadu_ps(&masked_input[i]);
            float w_signs[16];
            for (size_t j = 0; j < 16; ++j) {
                state = state * 1664525u + 1013904223u;
                float sign = (state & 1) ? 1.0f : -1.0f;
                w_signs[j] = static_cast<float>(ternary_weight[i + j]) * sign;
            }
            __m512 w_vec = _mm512_loadu_ps(w_signs);
            acc = _mm512_fmadd_ps(m_in, w_vec, acc);
        }
        float total = _mm512_reduce_add_ps(acc);
        for (; i < dim; ++i) {
            state = state * 1664525u + 1013904223u;
            float sign = (state & 1) ? 1.0f : -1.0f;
            total += masked_input[i] * (static_cast<float>(ternary_weight[i]) * sign);
        }
        return total;
    }
};

} // namespace weft
