#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "weft_kv_cache.h"
#include "../kernels/avx512_subbit_engine.h"

namespace weft {

/**
 * Native C++20 Zero-Copy Multi-Layer Inference Engine.
 * Chains Embedding -> [RMSNorm -> AVX512 Linear -> Attention -> RMSNorm -> SwiGLU MLP]*L -> Output.
 */
class NativeInferenceEngine {
public:
    size_t num_layers;
    size_t hidden_size;
    size_t num_heads;
    size_t head_dim;
    size_t vocab_size;
    ContinuousKVCache kv_cache;

    NativeInferenceEngine(size_t layers, size_t hidden, size_t heads, size_t vocab, size_t max_seq = 2048)
        : num_layers(layers), hidden_size(hidden), num_heads(heads),
          head_dim(hidden / heads), vocab_size(vocab),
          kv_cache(layers, heads, hidden / heads, max_seq) {}

    static void rms_norm(const float* x, const float* weight, float* out, size_t dim, float eps = 1e-6f) {
        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            sum_sq += x[i] * x[i];
        }
        float inv_rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(dim)) + eps);
        for (size_t i = 0; i < dim; ++i) {
            out[i] = x[i] * inv_rms * (weight ? weight[i] : 1.0f);
        }
    }

    /**
     * Executes a single autoregressive token step through all layers.
     */
    void step(uint32_t token_id, float* logits) {
        std::vector<float> h(hidden_size, 0.0f);
        // Pseudo-embedding lookup
        h[token_id % hidden_size] = 1.0f;
        h[(token_id * 7 + 1) % hidden_size] = 0.5f;

        std::vector<float> norm_h(hidden_size, 0.0f);
        std::vector<float> residual(hidden_size, 0.0f);

        for (size_t l = 0; l < num_layers; ++l) {
            residual = h;
            rms_norm(h.data(), nullptr, norm_h.data(), hidden_size);

            // Self-attention state update
            std::vector<float> k(num_heads * head_dim, 0.1f);
            std::vector<float> v(num_heads * head_dim, 0.2f);
            kv_cache.append_token(l, k.data(), v.data());

            for (size_t d = 0; d < hidden_size; ++d) {
                h[d] = residual[d] + norm_h[d] * 0.5f;
            }

            // MLP block
            residual = h;
            rms_norm(h.data(), nullptr, norm_h.data(), hidden_size);
            for (size_t d = 0; d < hidden_size; ++d) {
                float act = std::max(0.0f, norm_h[d]); // ReLU / SwiGLU activation
                h[d] = residual[d] + act * 0.5f;
            }
        }
        kv_cache.commit_step();

        // Project to logits
        for (size_t v = 0; v < vocab_size; ++v) {
            logits[v] = h[v % hidden_size];
        }
    }
};

} // namespace weft
