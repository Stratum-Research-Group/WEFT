#pragma once
// weft_tropical_toric_ssm.h
// Milestone Paper 262: Asymmetric Tropical Toric Ideals & Graded Gröbner Bases
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized tropical SSM recurrence step:
// out_state[i] = max(hidden[i] + a, input[i] + b)
inline void avx512_tropical_ssm_step(
    const float* __restrict__ hidden,
    const float* __restrict__ input,
    float* __restrict__ out_state,
    size_t count,
    float a,
    float b
) {
    __m512 va = _mm512_set1_ps(a);
    __m512 vb = _mm512_set1_ps(b);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vh = _mm512_loadu_ps(hidden + i);
        __m512 vx = _mm512_loadu_ps(input + i);

        __m512 v_ha = _mm512_add_ps(vh, va);
        __m512 v_xb = _mm512_add_ps(vx, vb);
        __m512 v_out = _mm512_max_ps(v_ha, v_xb);

        _mm512_storeu_ps(out_state + i, v_out);
    }

    for (; i < count; ++i) {
        float ha = hidden[i] + a;
        float xb = input[i] + b;
        out_state[i] = (ha > xb) ? ha : xb;
    }
}

} // namespace weft
