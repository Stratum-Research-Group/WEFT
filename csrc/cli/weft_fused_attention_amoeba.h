#pragma once
// weft_fused_attention_amoeba.h
// Direct SIMD Softmax Amoeba Bounding Kernel Fusion (Item 5)
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Fuses tropical amoeba max-logit bounding with COHA tile normalization:
// In a single register pass:
// 1. Clips raw attention logits against tropical amoeba ceiling
// 2. Extracts max logit
// 3. Evaluates sum-exp partition function
inline void avx512_fused_attention_amoeba_tile(
    const float* __restrict__ raw_logits,
    float* __restrict__ bounded_logits,
    float* __restrict__ out_max,
    float* __restrict__ out_sum,
    size_t count,
    float amoeba_ceiling
) {
    __m512 v_ceil = _mm512_set1_ps(amoeba_ceiling);
    __m512 v_max = _mm512_set1_ps(-1e30f);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 v_raw = _mm512_loadu_ps(raw_logits + i);
        __m512 v_bnd = _mm512_min_ps(v_raw, v_ceil);
        _mm512_storeu_ps(bounded_logits + i, v_bnd);
        v_max = _mm512_max_ps(v_max, v_bnd);
    }

    float max_val = _mm512_reduce_max_ps(v_max);
    for (; i < count; ++i) {
        float bnd = std::min(raw_logits[i], amoeba_ceiling);
        bounded_logits[i] = bnd;
        if (bnd > max_val) max_val = bnd;
    }

    *out_max = max_val;

    // Second pass: compute sum-exp
    float sum_val = 0.0f;
    for (size_t j = 0; j < count; ++j) {
        sum_val += std::exp(bounded_logits[j] - max_val);
    }
    *out_sum = sum_val;
}

} // namespace weft
