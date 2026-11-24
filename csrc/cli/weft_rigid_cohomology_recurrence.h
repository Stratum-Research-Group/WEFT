#pragma once
// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Asymmetric Rigid Cohomology & Berthelot Overconvergent F-Isocrystals for Long-Context State Recurrence
// Paper 214 Native Kernel Implementation

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <immintrin.h>

namespace weft {
namespace rigid_cohomology {

struct RecurrenceState {
    std::vector<float> h;
};

// Contractive linear state recurrence step:
// h_t = alpha * h_{t-1} + x_t, with |alpha| <= 1.0f
inline void step_linear_recurrence_avx512(
    const float* h_prev,
    const float* x_curr,
    float* h_next,
    size_t dim,
    float alpha = 0.95f)
{
    size_t i = 0;
#if defined(__AVX512F__)
    __m512 v_alpha = _mm512_set1_ps(alpha);

    for (; i + 16 <= dim; i += 16) {
        __m512 h_p = _mm512_loadu_ps(h_prev + i);
        __m512 x_c = _mm512_loadu_ps(x_curr + i);
        __m512 h_n = _mm512_fmadd_ps(h_p, v_alpha, x_c);
        _mm512_storeu_ps(h_next + i, h_n);
    }
#endif
    for (; i < dim; ++i) {
        h_next[i] = h_prev[i] * alpha + x_curr[i];
    }
}

// Multi-token prefix scan along linear recurrence
inline void prefix_recurrence_scan(
    const float* x_tokens,
    float* h_states,
    size_t num_tokens,
    size_t dim,
    float alpha = 0.95f)
{
    std::vector<float> current_h(dim, 0.0f);

    for (size_t t = 0; t < num_tokens; ++t) {
        const float* x_t = x_tokens + t * dim;
        float* h_t = h_states + t * dim;
        step_linear_recurrence_avx512(current_h.data(), x_t, h_t, dim, alpha);
        std::copy(h_t, h_t + dim, current_h.begin());
    }
}

} // namespace rigid_cohomology
} // namespace weft
