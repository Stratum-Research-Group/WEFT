#pragma once
// weft_tropical_wronskian_pruning.h
// Milestone Paper 253: Asymmetric Tropical Wronskian Determinants & Metric Graph Linear Series
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Evaluates pairwise tropical Wronskian slope valuation over head activation representations:
// out_val[i] = |head_a[i] - head_b[i]| using _mm512_abs_ps (AVX-512F)
// and computes max slope difference
inline float avx512_tropical_wronskian_valuation(
    const float* __restrict__ head_a,
    const float* __restrict__ head_b,
    float* __restrict__ out_diff,
    size_t count
) {
    size_t i = 0;
    __m512 v_max_val = _mm512_setzero_ps();

    for (; i + 16 <= count; i += 16) {
        __m512 va = _mm512_loadu_ps(head_a + i);
        __m512 vb = _mm512_loadu_ps(head_b + i);

        __m512 vdiff = _mm512_sub_ps(va, vb);
        __m512 vabs = _mm512_abs_ps(vdiff);

        _mm512_storeu_ps(out_diff + i, vabs);
        v_max_val = _mm512_max_ps(v_max_val, vabs);
    }

    float max_val = _mm512_reduce_max_ps(v_max_val);

    for (; i < count; ++i) {
        float diff = std::fabs(head_a[i] - head_b[i]);
        out_diff[i] = diff;
        if (diff > max_val) {
            max_val = diff;
        }
    }

    return max_val;
}

} // namespace weft
