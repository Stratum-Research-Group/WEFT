#pragma once
// weft_derived_kodaira_spencer.h
// Milestone Paper 246: Native AVX-512 Derived Kodaira-Spencer Dynamic Layer Pruning
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Compute relative deformation energy: E = sum(delta^2) / (sum(x^2) + eps)
// Returns true if layer should be skipped (E < tau_ks).
inline bool avx512_should_skip_layer_ks(
    const float* __restrict__ in_x,
    const float* __restrict__ delta_x,
    size_t count,
    float tau_ks
) {
    __m512 v_sum_norm = _mm512_setzero_ps();
    __m512 v_sum_delta = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vx = _mm512_loadu_ps(in_x + i);
        __m512 vd = _mm512_loadu_ps(delta_x + i);

        v_sum_norm = _mm512_fmadd_ps(vx, vx, v_sum_norm);
        v_sum_delta = _mm512_fmadd_ps(vd, vd, v_sum_delta);
    }

    float norm_sq = _mm512_reduce_add_ps(v_sum_norm);
    float delta_sq = _mm512_reduce_add_ps(v_sum_delta);

    for (; i < count; ++i) {
        norm_sq += in_x[i] * in_x[i];
        delta_sq += delta_x[i] * delta_x[i];
    }

    float energy = delta_sq / (norm_sq + 1e-6f);
    return energy < tau_ks;
}

} // namespace weft
