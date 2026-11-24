#pragma once
// weft_tropical_compactification_mask.h
// Milestone Paper 238: Native AVX-512 Tropical Compactification for Sparse Attention
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Generate sparse attention 16-bit bitmasks using AVX-512:
// mask bit is 1 if score[i] >= threshold_tau, else 0.
inline void avx512_tropical_sparse_mask(
    const float* __restrict__ scores,
    uint16_t* __restrict__ out_masks,
    size_t count_tokens,
    float threshold_tau
) {
    __m512 v_tau = _mm512_set1_ps(threshold_tau);
    size_t num_blocks = count_tokens / 16;

    for (size_t b = 0; b < num_blocks; ++b) {
        __m512 v_s = _mm512_loadu_ps(scores + b * 16);
        __mmask16 mask = _mm512_cmp_ps_mask(v_s, v_tau, _CMP_GE_OQ);
        out_masks[b] = static_cast<uint16_t>(mask);
    }
}

} // namespace weft
