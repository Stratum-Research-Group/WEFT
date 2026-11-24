#pragma once
// weft_derived_taq_activation.h
// Milestone Paper 258: Asymmetric Derived Ring Spectra & Topological André-Quillen Cohomology
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized piece-wise linear activation (TAQ approximation):
// out[i] = slope * in[i] + bias, clamped to min_val
inline void avx512_taq_linear_activation(
    const float* __restrict__ in_vec,
    float* __restrict__ out_vec,
    size_t count,
    float slope,
    float bias,
    float min_val
) {
    __m512 v_slope = _mm512_set1_ps(slope);
    __m512 v_bias = _mm512_set1_ps(bias);
    __m512 v_min = _mm512_set1_ps(min_val);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vx = _mm512_loadu_ps(in_vec + i);
        __m512 vy = _mm512_fmadd_ps(v_slope, vx, v_bias);
        vy = _mm512_max_ps(vy, v_min);
        _mm512_storeu_ps(out_vec + i, vy);
    }

    for (; i < count; ++i) {
        float y = slope * in_vec[i] + bias;
        out_vec[i] = (y < min_val) ? min_val : y;
    }
}

} // namespace weft
