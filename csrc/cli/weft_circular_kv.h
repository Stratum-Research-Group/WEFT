#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "weft_paged_kv.h"

namespace weft {

class WeftCircularKVCache {
public:
    size_t dim;
    size_t max_blocks;
    size_t tokens_per_block;
    size_t max_tokens;
    std::vector<WeftKVBlock> ring_blocks;
    size_t total_tokens_appended{0};

    WeftCircularKVCache(size_t dim, size_t max_blocks = 128, size_t tokens_per_block = WEFT_KV_PAGE_TOKENS)
        : dim(dim), max_blocks(max_blocks), tokens_per_block(tokens_per_block), max_tokens(max_blocks * tokens_per_block) {
        // Pre-allocate all circular blocks to guarantee ZERO dynamic allocations in token loop
        ring_blocks.resize(max_blocks);
        for (auto& b : ring_blocks) {
            b.scale = 1.0f;
            b.zero_point = 0.0f;
            b.num_tokens = 0;
            b.flags = 0;
            b.packed_keys.resize((tokens_per_block * dim + 3) / 4, 0);
            b.packed_values.resize((tokens_per_block * dim + 3) / 4, 0);
        }
    }

    // Append token with zero dynamic heap allocations
    void append_token_zero_alloc(const float* key_vec, const float* val_vec) {
        size_t block_idx = (total_tokens_appended / tokens_per_block) % max_blocks;
        size_t token_offset = total_tokens_appended % tokens_per_block;

        WeftKVBlock& current = ring_blocks[block_idx];
        if (token_offset == 0) {
            current.num_tokens = 0;
            std::fill(current.packed_keys.begin(), current.packed_keys.end(), 0);
            std::fill(current.packed_values.begin(), current.packed_values.end(), 0);
        }

        float k_min = key_vec[0], k_max = key_vec[0];
        for (size_t i = 1; i < dim; ++i) {
            if (key_vec[i] < k_min) k_min = key_vec[i];
            if (key_vec[i] > k_max) k_max = key_vec[i];
        }

        float s = (k_max - k_min > 1e-6f) ? (k_max - k_min) / 3.0f : 1.0f;
        float z = -k_min / s;
        current.scale = s;
        current.zero_point = z;

        for (size_t i = 0; i < dim; ++i) {
            int q = std::clamp(static_cast<int>(std::round(key_vec[i] / s + z)), 0, 3);
            size_t bit_idx = (token_offset * dim + i) * 2;
            size_t byte_idx = bit_idx / 8;
            size_t shift = bit_idx % 8;
            current.packed_keys[byte_idx] |= (static_cast<uint8_t>(q) & 0x03) << shift;
        }

        current.num_tokens++;
        total_tokens_appended++;
    }

    size_t active_tokens() const {
        return std::min(total_tokens_appended, max_tokens);
    }
};

} // namespace weft
