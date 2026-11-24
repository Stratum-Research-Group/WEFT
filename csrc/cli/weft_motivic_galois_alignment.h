#pragma once
// weft_motivic_galois_alignment.h
// Milestone Paper 263: Asymmetric Motivic Galois Groups & Tannakian Categories
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized Tannakian multimodal alignment blend:
// out_aligned[i] = w1 * mod1[i] + w2 * mod2[i]
inline void avx512_tannakian_alignment_blend(
    const float* __restrict__ mod1,
    const float* __restrict__ mod2,
    float* __restrict__ out_aligned,
    size_t count,
    float w1,
    float w2
) {
    __m512 vw1 = _mm512_set1_ps(w1);
    __m512 vw2 = _mm512_set1_ps(w2);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 v1 = _mm512_loadu_ps(mod1 + i);
        __m512 v2 = _mm512_loadu_ps(mod2 + i);
        __m512 v_blend = _mm512_fmadd_ps(vw1, v1, _mm512_mul_ps(vw2, v2));
        _mm512_storeu_ps(out_aligned + i, v_blend);
    }

    for (; i < count; ++i) {
        out_aligned[i] = w1 * mod1[i] + w2 * mod2[i];
    }
}

} // namespace weft
