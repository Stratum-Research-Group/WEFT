#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

namespace weft {

// DeepSeek MLA Parameters:
// Compressed latent dim: 512, Decoupled RoPE dim: 64 -> Total latent KV dim = 576
constexpr size_t MLA_LATENT_KV_DIM = 576;
constexpr size_t MLA_PAGE_TOKENS = 16;

struct alignas(32) WeftMLAKVBlock {
    float scale;
    float zero_point;
    uint16_t num_tokens;
    uint16_t active_mask; // Bitmask of valid tree nodes in this block
    uint8_t flags;
    uint8_t reserved[3];
    // 576 dims * 2 bits = 1152 bits = 144 bytes per token!
    // 16 tokens * 144 bytes = 2304 bytes per block
    std::vector<uint8_t> packed_latent_kv;
};

class WeftMLAPagedKVCache {
public:
    size_t num_layers;

    WeftMLAPagedKVCache(size_t num_layers) : num_layers(num_layers) {}

    // Store compressed MLA latent KV vector (576 dims)
    void append_mla_token(const float* latent_kv_576, std::vector<WeftMLAKVBlock>& blocks) {
        if (blocks.empty() || blocks.back().num_tokens >= MLA_PAGE_TOKENS) {
            WeftMLAKVBlock new_block;
            new_block.scale = 1.0f;
            new_block.zero_point = 0.0f;
            new_block.num_tokens = 0;
            new_block.active_mask = 0xFFFF;
            new_block.flags = 0;
            new_block.packed_latent_kv.resize(MLA_PAGE_TOKENS * 144, 0);
            blocks.push_back(std::move(new_block));
        }

        WeftMLAKVBlock& current = blocks.back();
        size_t token_offset = current.num_tokens;

        float v_min = latent_kv_576[0], v_max = latent_kv_576[0];
        for (size_t i = 1; i < MLA_LATENT_KV_DIM; ++i) {
            if (latent_kv_576[i] < v_min) v_min = latent_kv_576[i];
            if (latent_kv_576[i] > v_max) v_max = latent_kv_576[i];
        }

        float s = (v_max - v_min > 1e-6f) ? (v_max - v_min) / 3.0f : 1.0f;
        float z = -v_min / s;
        current.scale = s;
        current.zero_point = z;

        // Pack 576 2-bit values into 144 bytes
        size_t base_byte = token_offset * 144;
        for (size_t i = 0; i < MLA_LATENT_KV_DIM; ++i) {
            int q = std::clamp(static_cast<int>(std::round(latent_kv_576[i] / s + z)), 0, 3);
            size_t byte_idx = base_byte + (i / 4);
            size_t shift = (i % 4) * 2;
            current.packed_latent_kv[byte_idx] |= (static_cast<uint8_t>(q) & 0x03) << shift;
        }

        current.num_tokens++;
    }

    // Dynamic early-exit branch prune: deactivate child nodes when parent rejected
    void prune_branch(WeftMLAKVBlock& block, uint16_t reject_node_mask) {
        block.active_mask &= ~reject_node_mask;
    }
};

} // namespace weft
