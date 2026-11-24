#pragma once
// weft_motivic_lfunctions_precision.h
// Milestone Paper 254: Asymmetric Motivic L-Functions & Beilinson Regulators for Sub-Bit Precision Allocation
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized precision allocation:
// out_precision[i] = max(min_bits, base_bits + inv_beta * (regulator[i] - mean_regulator))
inline void avx512_motivic_precision_allocation(
    const float* __restrict__ regulators,
    float* __restrict__ out_precision,
    size_t count,
    float base_bits,
    float min_bits,
    float mean_regulator,
    float inv_beta
) {
    __m512 v_base = _mm512_set1_ps(base_bits);
    __m512 v_min = _mm512_set1_ps(min_bits);
    __m512 v_mean = _mm512_set1_ps(mean_regulator);
    __m512 v_inv_beta = _mm512_set1_ps(inv_beta);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 v_reg = _mm512_loadu_ps(regulators + i);
        __m512 v_diff = _mm512_sub_ps(v_reg, v_mean);
        __m512 v_alloc = _mm512_fmadd_ps(v_inv_beta, v_diff, v_base);
        v_alloc = _mm512_max_ps(v_alloc, v_min);

        _mm512_storeu_ps(out_precision + i, v_alloc);
    }

    for (; i < count; ++i) {
        float alloc = base_bits + inv_beta * (regulators[i] - mean_regulator);
        if (alloc < min_bits) {
            alloc = min_bits;
        }
        out_precision[i] = alloc;
    }
}

} // namespace weft
