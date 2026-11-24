#pragma once
// weft_sparse_attention.h - Asymmetric Sub-Bit Attention Pruning & Dynamic Sparsification
// Milestone Paper #123
// Authors: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

// Represents a 2-bit compressed KV block with coarse centroid statistics
struct SparseKVBlock {
    static constexpr size_t BLOCK_SIZE = 16;
    static constexpr size_t HEAD_DIM = 64;
    static constexpr size_t PACKED_BYTES = (BLOCK_SIZE * HEAD_DIM * 2) / 8; // 256 bytes

    uint32_t block_id{0};
    uint16_t num_tokens{BLOCK_SIZE};
    float centroid_dot{0.0f};      // q^T * c_b coarse dot-product
    float block_radius{0.0f};      // Maximum residual deviation in block
    float scale{1.0f};
    float zero_point{0.0f};
    
    // Coarse centroid vector for fast screening (dimension = HEAD_DIM)
    alignas(64) float centroid[HEAD_DIM];

    // 2-bit packed key & value representations
    alignas(64) uint8_t packed_keys[PACKED_BYTES];
    alignas(64) uint8_t packed_values[PACKED_BYTES];
};

struct AttentionSparsityConfig {
    float threshold_delta{0.01f}; // Top-p / threshold relative tolerance
    float min_pruning_ratio{0.50f}; // Minimum target sparsity
    bool enable_avx512{true};
};

class WeftSparseAttentionEngine {
public:
    explicit WeftSparseAttentionEngine(AttentionSparsityConfig config = {})
        : config_(config) {}

    // Screen blocks and generate active sparse index mask
    // Returns number of active blocks selected
    size_t screen_blocks(
        const float* query,
        const SparseKVBlock* blocks,
        size_t total_blocks,
        uint32_t* active_block_indices,
        float& out_max_logit
    ) {
        if (total_blocks == 0) return 0;

        std::vector<float> block_upper_bounds(total_blocks, 0.0f);
        float global_max = -1e9f;

        // Compute coarse centroid upper bounds
        for (size_t b = 0; b < total_blocks; ++b) {
            float dot = 0.0f;
#if defined(__AVX512F__)
            if (config_.enable_avx512) {
                __m512 vdot = _mm512_setzero_ps();
                for (size_t d = 0; d < SparseKVBlock::HEAD_DIM; d += 16) {
                    __m512 vq = _mm512_loadu_ps(query + d);
                    __m512 vc = _mm512_loadu_ps(blocks[b].centroid + d);
                    vdot = _mm512_fmadd_ps(vq, vc, vdot);
                }
                dot = _mm512_reduce_add_ps(vdot);
            } else {
#endif
                for (size_t d = 0; d < SparseKVBlock::HEAD_DIM; ++d) {
                    dot += query[d] * blocks[b].centroid[d];
                }
#if defined(__AVX512F__)
            }
#endif
            float scale_factor = 1.0f / std::sqrt(static_cast<float>(SparseKVBlock::HEAD_DIM));
            float upper_bound = (dot + blocks[b].block_radius) * scale_factor;
            block_upper_bounds[b] = upper_bound;
            if (upper_bound > global_max) {
                global_max = upper_bound;
            }
        }

        out_max_logit = global_max;
        float pruning_threshold = global_max - std::log(1.0f / config_.threshold_delta);

        // Filter active blocks
        size_t active_count = 0;
        for (size_t b = 0; b < total_blocks; ++b) {
            if (block_upper_bounds[b] >= pruning_threshold) {
                active_block_indices[active_count++] = static_cast<uint32_t>(b);
            }
        }

        // Guarantee at least 1 active block (e.g. the maximum)
        if (active_count == 0 && total_blocks > 0) {
            active_block_indices[0] = 0;
            active_count = 1;
        }

        return active_count;
    }

    // Compute sparse attention over selected active blocks
    void compute_sparse_attention(
        const float* query,
        const SparseKVBlock* blocks,
        const uint32_t* active_block_indices,
        size_t active_count,
        float* out_attention_values
    ) {
        if (active_count == 0) return;

        // Reset output accumulation
        for (size_t d = 0; d < SparseKVBlock::HEAD_DIM; ++d) {
            out_attention_values[d] = 0.0f;
        }

        float normalizer = 0.0f;
        std::vector<float> token_scores;
        token_scores.reserve(active_count * SparseKVBlock::BLOCK_SIZE);

        float max_score = -1e9f;

        // Phase 1: Compute token dot products for active blocks
        for (size_t idx = 0; idx < active_count; ++idx) {
            uint32_t b = active_block_indices[idx];
            const auto& blk = blocks[b];

            for (size_t t = 0; t < blk.num_tokens; ++t) {
                float score = 0.0f;
                // Unpack 2-bit key on the fly
                for (size_t d = 0; d < SparseKVBlock::HEAD_DIM; ++d) {
                    size_t bit_offset = (t * SparseKVBlock::HEAD_DIM + d) * 2;
                    size_t byte_idx = bit_offset / 8;
                    size_t shift = bit_offset % 8;
                    uint8_t q_val = (blk.packed_keys[byte_idx] >> shift) & 0x03;
                    float k_val = static_cast<float>(q_val) * blk.scale + blk.zero_point;
                    score += query[d] * k_val;
                }
                score /= std::sqrt(static_cast<float>(SparseKVBlock::HEAD_DIM));
                token_scores.push_back(score);
                if (score > max_score) {
                    max_score = score;
                }
            }
        }

        // Phase 2: Exponentiate and accumulate
        size_t token_cursor = 0;
        for (size_t idx = 0; idx < active_count; ++idx) {
            uint32_t b = active_block_indices[idx];
            const auto& blk = blocks[b];

            for (size_t t = 0; t < blk.num_tokens; ++t) {
                float exp_val = std::exp(token_scores[token_cursor++] - max_score);
                normalizer += exp_val;

                // Accumulate weighted values
                for (size_t d = 0; d < SparseKVBlock::HEAD_DIM; ++d) {
                    size_t bit_offset = (t * SparseKVBlock::HEAD_DIM + d) * 2;
                    size_t byte_idx = bit_offset / 8;
                    size_t shift = bit_offset % 8;
                    uint8_t q_val = (blk.packed_values[byte_idx] >> shift) & 0x03;
                    float v_val = static_cast<float>(q_val) * blk.scale + blk.zero_point;
                    out_attention_values[d] += exp_val * v_val;
                }
            }
        }

        // Phase 3: Final Softmax normalization
        if (normalizer > 1e-8f) {
            float inv_norm = 1.0f / normalizer;
            for (size_t d = 0; d < SparseKVBlock::HEAD_DIM; ++d) {
                out_attention_values[d] *= inv_norm;
            }
        }
    }

private:
    AttentionSparsityConfig config_;
};

} // namespace weft
