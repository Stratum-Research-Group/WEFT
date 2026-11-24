#pragma once
// weft_derived_gauss_manin.h
// Milestone Paper 264: Asymmetric Derived De Rham Cohomology & Gauss-Manin Connections
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized Gauss-Manin parallel transport update:
// out_weights[i] = weights[i] - ds * (connection[i] * weights[i])
inline void avx512_gauss_manin_transport(
    const float* __restrict__ weights,
    const float* __restrict__ connection,
    float* __restrict__ out_weights,
    size_t count,
    float ds
) {
    __m512 v_ds = _mm512_set1_ps(ds);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vw = _mm512_loadu_ps(weights + i);
        __m512 vc = _mm512_loadu_ps(connection + i);
        __m512 v_drift = _mm512_mul_ps(vc, vw);
        __m512 v_out = _mm512_fnmadd_ps(v_ds, v_drift, vw);
        _mm512_storeu_ps(out_weights + i, v_out);
    }

    for (; i < count; ++i) {
        out_weights[i] = weights[i] - ds * connection[i] * weights[i];
    }
}

} // namespace weft
