#pragma once
// weft_derived_critical_loci.h
// Milestone Paper 261: Asymmetric Derived Critical Loci & Vanishing Cycles
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized vanishing cycle gradient smoothing:
// out_gradients[i] = raw_gradients[i] - gamma * vanishing_cycles[i]
inline void avx512_vanishing_cycle_smoothing(
    const float* __restrict__ raw_gradients,
    const float* __restrict__ vanishing_cycles,
    float* __restrict__ out_gradients,
    size_t count,
    float gamma
) {
    __m512 v_gamma = _mm512_set1_ps(gamma);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 v_raw = _mm512_loadu_ps(raw_gradients + i);
        __m512 v_cycle = _mm512_loadu_ps(vanishing_cycles + i);
        __m512 v_smooth = _mm512_fnmadd_ps(v_cycle, v_gamma, v_raw);
        _mm512_storeu_ps(out_gradients + i, v_smooth);
    }

    for (; i < count; ++i) {
        out_gradients[i] = raw_gradients[i] - gamma * vanishing_cycles[i];
    }
}

} // namespace weft
