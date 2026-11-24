#pragma once
// weft_derived_loop_speculative.h
// Milestone Paper 240: Native AVX-512 Derived Loop Speculative Verification
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <immintrin.h>
#include <vector>

namespace weft {

// Vectorized loop consistency verification:
// checks if |draft[i] - target[i]| <= tolerance for all tokens in draft window.
// Returns count of consecutive valid tokens before first failure.
inline size_t avx512_verify_speculative_loop(
    const float* __restrict__ draft_logits,
    const float* __restrict__ target_logits,
    size_t length,
    float tolerance
) {
    __m512 v_tol = _mm512_set1_ps(tolerance);
    size_t valid_count = 0;

    size_t i = 0;
    for (; i + 16 <= length; i += 16) {
        __m512 vd = _mm512_loadu_ps(draft_logits + i);
        __m512 vt = _mm512_loadu_ps(target_logits + i);
        __m512 v_diff = _mm512_abs_ps(_mm512_sub_ps(vd, vt));

        __mmask16 mask = _mm512_cmp_ps_mask(v_diff, v_tol, _CMP_LE_OQ);
        if (mask == 0xFFFF) {
            valid_count += 16;
        } else {
            // Find first failing bit
            uint32_t fail_idx = __builtin_ctz(~mask);
            valid_count += fail_idx;
            return valid_count;
        }
    }

    for (; i < length; ++i) {
        if (std::fabs(draft_logits[i] - target_logits[i]) <= tolerance) {
            valid_count++;
        } else {
            break;
        }
    }

    return valid_count;
}

} // namespace weft
