#pragma once
// weft_tropical_grassmannian_pruning.h
// Milestone Paper 265: Asymmetric Tropical Grassmannians & Plücker Coordinates
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized tropical Plücker relation discrepancy evaluation:
// out_diff[i] = |(pik[i] + pjl[i]) - (pil[i] + pjk[i])|
inline void avx512_tropical_plucker_discrepancy(
    const float* __restrict__ pik,
    const float* __restrict__ pjl,
    const float* __restrict__ pil,
    const float* __restrict__ pjk,
    float* __restrict__ out_diff,
    size_t count
) {
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 v_pik = _mm512_loadu_ps(pik + i);
        __m512 v_pjl = _mm512_loadu_ps(pjl + i);
        __m512 v_pil = _mm512_loadu_ps(pil + i);
        __m512 v_pjk = _mm512_loadu_ps(pjk + i);

        __m512 v_t1 = _mm512_add_ps(v_pik, v_pjl);
        __m512 v_t2 = _mm512_add_ps(v_pil, v_pjk);
        __m512 v_diff = _mm512_abs_ps(_mm512_sub_ps(v_t1, v_t2));

        _mm512_storeu_ps(out_diff + i, v_diff);
    }

    for (; i < count; ++i) {
        float t1 = pik[i] + pjl[i];
        float t2 = pil[i] + pjk[i];
        out_diff[i] = std::fabs(t1 - t2);
    }
}

} // namespace weft
