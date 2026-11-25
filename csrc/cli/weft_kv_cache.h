#pragma once

#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace weft {

/**
 * Ring-buffered continuous Key-Value cache for autoregressive decoding
 * with paged block allocation and speculative acceptance rollback.
 */
class ContinuousKVCache {
public:
    size_t num_layers;
    size_t num_heads;
    size_t head_dim;
    size_t max_seq_len;
    size_t current_len;

    // Flat storage per layer: [2 (K, V), num_heads, max_seq_len, head_dim]
    std::vector<std::vector<float>> layer_buffers;

    ContinuousKVCache(size_t layers, size_t heads, size_t dim, size_t max_seq)
        : num_layers(layers), num_heads(heads), head_dim(dim), max_seq_len(max_seq), current_len(0) {
        size_t layer_elements = 2 * num_heads * max_seq_len * head_dim;
        layer_buffers.resize(num_layers, std::vector<float>(layer_elements, 0.0f));
    }

    void reset() {
        current_len = 0;
        for (auto& buf : layer_buffers) {
            std::fill(buf.begin(), buf.end(), 0.0f);
        }
    }

    /**
     * Appends a new token's Key and Value slices across all heads for a layer.
     * k_src, v_src must each have size (num_heads * head_dim).
     */
    void append_token(size_t layer_idx, const float* k_src, const float* v_src) {
        if (layer_idx >= num_layers) {
            throw std::out_of_range("Layer index exceeds cache configuration.");
        }
        if (current_len >= max_seq_len) {
            throw std::runtime_error("KV cache capacity exceeded max_seq_len.");
        }

        float* base = layer_buffers[layer_idx].data();
        size_t kv_stride = num_heads * max_seq_len * head_dim;

        // Copy K
        for (size_t h = 0; h < num_heads; ++h) {
            size_t offset_k = (0 * kv_stride) + (h * max_seq_len * head_dim) + (current_len * head_dim);
            std::memcpy(base + offset_k, k_src + (h * head_dim), head_dim * sizeof(float));
        }

        // Copy V
        for (size_t h = 0; h < num_heads; ++h) {
            size_t offset_v = (1 * kv_stride) + (h * max_seq_len * head_dim) + (current_len * head_dim);
            std::memcpy(base + offset_v, v_src + (h * head_dim), head_dim * sizeof(float));
        }
    }

    void commit_step() {
        current_len++;
    }

    /**
     * Roll back cache position when speculative tokens are rejected.
     */
    void rollback(size_t accepted_tokens) {
        current_len = std::min(current_len, accepted_tokens);
    }

    const float* get_k_ptr(size_t layer_idx, size_t head_idx) const {
        size_t kv_stride = num_heads * max_seq_len * head_dim;
        return layer_buffers[layer_idx].data() + (0 * kv_stride) + (head_idx * max_seq_len * head_dim);
    }

    const float* get_v_ptr(size_t layer_idx, size_t head_idx) const {
        size_t kv_stride = num_heads * max_seq_len * head_dim;
        return layer_buffers[layer_idx].data() + (1 * kv_stride) + (head_idx * max_seq_len * head_dim);
    }
};

} // namespace weft
