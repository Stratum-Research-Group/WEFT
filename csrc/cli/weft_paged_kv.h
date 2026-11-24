#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace weft {

// Block page size: 16 tokens per block
constexpr size_t WEFT_KV_PAGE_TOKENS = 16;

// 2-bit quantized KV page structure (32-byte cache aligned)
struct alignas(32) WeftKVBlock {
    float scale;         // Local block scale factor
    float zero_point;    // Local block zero point
    uint16_t num_tokens; // Number of stored tokens in this block
    uint8_t flags;       // Allocation status & state flags
    uint8_t reserved[5]; // Alignment padding to ensure 16-byte boundary
    // Packed 2-bit values: 4 values per byte
    std::vector<uint8_t> packed_keys;
    std::vector<uint8_t> packed_values;
};

class WeftPagedKVCache {
public:
    size_t dim;
    size_t num_layers;
    size_t num_heads;
    size_t head_dim;

    WeftPagedKVCache(size_t dim, size_t num_layers, size_t num_heads)
        : dim(dim), num_layers(num_layers), num_heads(num_heads), head_dim(dim / num_heads) {}

    // Quantize and store a sequence of key/value activations into block-paged structure
    void append_token(size_t layer_idx, const float* key_vec, const float* val_vec, std::vector<WeftKVBlock>& blocks) {
        if (blocks.empty() || blocks.back().num_tokens >= WEFT_KV_PAGE_TOKENS) {
            WeftKVBlock new_block;
            new_block.scale = 1.0f;
            new_block.zero_point = 0.0f;
            new_block.num_tokens = 0;
            new_block.flags = 0;
            new_block.packed_keys.resize((WEFT_KV_PAGE_TOKENS * dim + 3) / 4, 0);
            new_block.packed_values.resize((WEFT_KV_PAGE_TOKENS * dim + 3) / 4, 0);
            blocks.push_back(std::move(new_block));
        }

        WeftKVBlock& current = blocks.back();
        size_t token_offset = current.num_tokens;

        // Compute local min/max for scale and zero-point
        float k_min = key_vec[0], k_max = key_vec[0];
        for (size_t i = 1; i < dim; ++i) {
            if (key_vec[i] < k_min) k_min = key_vec[i];
            if (key_vec[i] > k_max) k_max = key_vec[i];
        }

        float s = (k_max - k_min > 1e-6f) ? (k_max - k_min) / 3.0f : 1.0f;
        float z = -k_min / s;
        current.scale = s;
        current.zero_point = z;

        // Pack 2-bit keys
        for (size_t i = 0; i < dim; ++i) {
            int q = std::clamp(static_cast<int>(std::round(key_vec[i] / s + z)), 0, 3);
            size_t bit_idx = (token_offset * dim + i) * 2;
            size_t byte_idx = bit_idx / 8;
            size_t shift = bit_idx % 8;
            current.packed_keys[byte_idx] |= (static_cast<uint8_t>(q) & 0x03) << shift;
        }

        current.num_tokens++;
    }
};

} // namespace weft
