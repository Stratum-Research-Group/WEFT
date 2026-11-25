#pragma once
// weft_derived_floer_homology.h
// Milestone Paper 270: Asymmetric Derived Lagrangian Intersections & Floer Homology
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized Floer momentum-regularized weight update:
// out_weights[i] = weights[i] + eta * momentum[i] - gamma * floer_correction[i]
inline void avx512_floer_momentum_update(
    const float* __restrict__ weights,
    const float* __restrict__ momentum,
    const float* __restrict__ floer_corrections,
    float* __restrict__ out_weights,
    size_t count,
    float eta,
    float gamma
) {
    __m512 v_eta = _mm512_set1_ps(eta);
    __m512 v_gamma = _mm512_set1_ps(gamma);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vw = _mm512_loadu_ps(weights + i);
        __m512 vm = _mm512_loadu_ps(momentum + i);
        __m512 vf = _mm512_loadu_ps(floer_corrections + i);

        __m512 v_step = _mm512_fmadd_ps(v_eta, vm, vw);
        __m512 v_out = _mm512_fnmadd_ps(v_gamma, vf, v_step);

        _mm512_storeu_ps(out_weights + i, v_out);
    }

    for (; i < count; ++i) {
        out_weights[i] = weights[i] + eta * momentum[i] - gamma * floer_corrections[i];
    }
}

} // namespace weft
