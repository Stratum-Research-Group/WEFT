#pragma once
// weft_derived_symplectic_optimizer.h
// Milestone Paper 236: Native AVX-512 Derived Symplectic Leapfrog Optimizer Updates
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

struct SymplecticOptimizerConfig {
    float epsilon; // step size
    float damping; // momentum damping factor (1.0 = purely symplectic)
};

// Vectorized symplectic leapfrog position and momentum integration using AVX-512:
// p_{t+1/2} = p_t - (epsilon / 2) * grad_t
// q_{t+1}   = q_t + epsilon * p_{t+1/2}
// p_{t+1}   = p_{t+1/2} - (epsilon / 2) * grad_{t+1}
inline void avx512_symplectic_leapfrog_step(
    float* __restrict__ q,
    float* __restrict__ p,
    const float* __restrict__ grad,
    size_t count,
    float epsilon
) {
    float half_eps = 0.5f * epsilon;
    __m512 v_half_eps = _mm512_set1_ps(half_eps);
    __m512 v_eps = _mm512_set1_ps(epsilon);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 v_q = _mm512_loadu_ps(q + i);
        __m512 v_p = _mm512_loadu_ps(p + i);
        __m512 v_g = _mm512_loadu_ps(grad + i);

        // p_{t+1/2} = v_p - half_eps * v_g
        __m512 v_p_half = _mm512_fnmadd_ps(v_half_eps, v_g, v_p);

        // q_{t+1} = v_q + epsilon * v_p_half
        __m512 v_q_next = _mm512_fmadd_ps(v_eps, v_p_half, v_q);

        _mm512_storeu_ps(q + i, v_q_next);
        _mm512_storeu_ps(p + i, v_p_half);
    }

    for (; i < count; ++i) {
        float p_half = p[i] - half_eps * grad[i];
        q[i] += epsilon * p_half;
        p[i] = p_half;
    }
}

} // namespace weft
