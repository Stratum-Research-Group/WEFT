#pragma once
// weft_motivic_stack_pruning.h
// Milestone Paper 272: Asymmetric Motivic Cohomological Operations on Deligne-Mumford Stacks
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace weft {

// Vectorized token stack-inertia pruning mask computation:
// out_retained[i] = 1 if inertia[i] >= threshold else 0
inline size_t avx512_stack_token_pruning(
    const float* __restrict__ inertia_vals,
    uint8_t* __restrict__ out_retained,
    size_t count,
    float threshold
) {
    size_t retained_count = 0;
    __m512 v_thresh = _mm512_set1_ps(threshold);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 v_val = _mm512_loadu_ps(inertia_vals + i);
        __mmask16 mask = _mm512_cmp_ps_mask(v_val, v_thresh, _CMP_GE_OQ);

        for (int b = 0; b < 16; ++b) {
            uint8_t ret = (mask >> b) & 1;
            out_retained[i + b] = ret;
            retained_count += ret;
        }
    }

    for (; i < count; ++i) {
        uint8_t ret = (inertia_vals[i] >= threshold) ? 1 : 0;
        out_retained[i] = ret;
        retained_count += ret;
    }

    return retained_count;
}

} // namespace weft
