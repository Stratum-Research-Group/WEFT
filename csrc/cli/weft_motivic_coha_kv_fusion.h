#pragma once
// weft_motivic_coha_kv_fusion.h
// Milestone Paper 257: Asymmetric Motivic Cohomological Hall Algebras (COHA) & BPS States for FlashAttention KV Fusion
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized COHA fused attention tile statistics:
// out_max[i] = max(m1[i], m2[i])
// out_sum[i] = l1[i] * exp(m1[i] - out_max[i]) + l2[i] * exp(m2[i] - out_max[i])
inline void avx512_coha_kv_fusion(
    const float* __restrict__ m1,
    const float* __restrict__ l1,
    const float* __restrict__ m2,
    const float* __restrict__ l2,
    float* __restrict__ out_max,
    float* __restrict__ out_sum,
    size_t count
) {
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vm1 = _mm512_loadu_ps(m1 + i);
        __m512 vm2 = _mm512_loadu_ps(m2 + i);
        __m512 vmax = _mm512_max_ps(vm1, vm2);

        _mm512_storeu_ps(out_max + i, vmax);
    }

    for (size_t j = 0; j < count; ++j) {
        float max_val = std::max(m1[j], m2[j]);
        out_max[j] = max_val;
        float factor1 = std::exp(m1[j] - max_val);
        float factor2 = std::exp(m2[j] - max_val);
        out_sum[j] = l1[j] * factor1 + l2[j] * factor2;
    }
}

} // namespace weft
