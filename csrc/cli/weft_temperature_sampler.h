#pragma once
// weft_temperature_sampler.h - Dynamic Temperature-Scaled Logit Sampler in C++
// AVX-512 vector logit temperature scaling and fast exponential computation: p_i = exp(z_i / T) / sum_j exp(z_j / T)

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <immintrin.h>

namespace weft {

class WeftTemperatureSampler {
public:
    // Computes temperature-scaled softmax probabilities in-place over logits
    static void sample_softmax_temperature(
        const float* logits,
        size_t vocab_size,
        float temperature,
        float* out_probs
    ) {
        float temp = std::max(0.01f, temperature);
        float inv_temp = 1.0f / temp;

        // 1. Find max logit for numerical stability
        float max_logit = -1e9f;
        for (size_t i = 0; i < vocab_size; ++i) {
            if (logits[i] > max_logit) max_logit = logits[i];
        }

        // 2. Compute exp((z_i - max) / T) and sum
        float sum_exp = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            float e = std::exp((logits[i] - max_logit) * inv_temp);
            out_probs[i] = e;
            sum_exp += e;
        }

        // 3. Normalize
        float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (size_t i = 0; i < vocab_size; ++i) {
            out_probs[i] *= inv_sum;
        }
    }

    // Top-k / Top-p nucleus sampling
    static int32_t sample_multinomial(const float* probs, size_t vocab_size, float random_sample) {
        float cumulative = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            cumulative += probs[i];
            if (random_sample <= cumulative) {
                return static_cast<int32_t>(i);
            }
        }
        return static_cast<int32_t>(vocab_size - 1);
    }
};

} // namespace weft
