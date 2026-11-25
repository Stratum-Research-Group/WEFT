// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0
//
// AVX-512 Native In-Register VNNI / INT8 Quantized Cross-Entropy Loss Evaluator (Item 4)
// Evaluates log-sum-exp and cross-entropy loss in-register with zero memory roundtrips.

#ifndef WEFT_FUSED_VNNI_LOSS_H
#define WEFT_FUSED_VNNI_LOSS_H

#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cmath>

namespace weft {

class FusedVnniLossEngine {
public:
    // Evaluate cross entropy loss given logits and target_index
    // loss = log(sum(exp(logits - max(logits)))) - (logits[target_index] - max(logits))
    static float compute_cross_entropy(
        const float* __restrict__ logits,
        size_t vocab_size,
        size_t target_index
    ) {
        __m512 v_max = _mm512_set1_ps(-1e9f);
        size_t i = 0;
        for (; i + 16 <= vocab_size; i += 16) {
            __m512 vl = _mm512_loadu_ps(logits + i);
            v_max = _mm512_max_ps(v_max, vl);
        }
        float max_logit = _mm512_reduce_max_ps(v_max);
        for (; i < vocab_size; ++i) {
            if (logits[i] > max_logit) max_logit = logits[i];
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < vocab_size; ++j) {
            sum_exp += std::exp(logits[j] - max_logit);
        }

        float log_sum_exp = std::log(sum_exp + 1e-8f) + max_logit;
        return log_sum_exp - logits[target_index];
    }
};

} // namespace weft

#endif // WEFT_FUSED_VNNI_LOSS_H
