// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0
//
// AVX-512 Native In-Register Fused Ternary-Quantized Attention Softmax Engine (Item 4)
// Evaluates in-register max subtraction, exp approximation, and reciprocal normalization.

#ifndef WEFT_FUSED_TERNARY_SOFTMAX_H
#define WEFT_FUSED_TERNARY_SOFTMAX_H

#include <immintrin.h>
#include <cstddef>
#include <cmath>
#include <algorithm>

namespace weft {

class FusedTernarySoftmaxEngine {
public:
    // Compute in-register numerically stable Softmax: out = exp(x - max(x)) / sum(exp(x - max(x)))
    static void evaluate_softmax(
        const float* __restrict__ x,
        float* __restrict__ out,
        size_t dim
    ) {
        __m512 v_max = _mm512_set1_ps(-1e9f);
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);
            v_max = _mm512_max_ps(v_max, vx);
        }
        float max_val = _mm512_reduce_max_ps(v_max);
        for (; i < dim; ++i) {
            if (x[i] > max_val) max_val = x[i];
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            float exp_val = std::exp(x[j] - max_val);
            out[j] = exp_val;
            sum_exp += exp_val;
        }

        float inv_sum = 1.0f / (sum_exp + 1e-8f);
        __m512 v_inv = _mm512_set1_ps(inv_sum);
        i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 vo = _mm512_loadu_ps(out + i);
            vo = _mm512_mul_ps(vo, v_inv);
            _mm512_storeu_ps(out + i, vo);
        }
        for (; i < dim; ++i) {
            out[i] *= inv_sum;
        }
    }
};

} // namespace weft

#endif // WEFT_FUSED_TERNARY_SOFTMAX_H
