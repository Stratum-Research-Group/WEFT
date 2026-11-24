// test_sparse_attention_pruning.cpp - Verification for Paper #123
#include "csrc/cli/weft_sparse_attention.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <random>

using namespace weft;

int main() {
    std::cout << "[TEST] Initializing test_sparse_attention_pruning..." << std::endl;

    constexpr size_t NUM_BLOCKS = 32; // 32 blocks * 16 tokens = 512 tokens
    std::vector<SparseKVBlock> blocks(NUM_BLOCKS);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> query(SparseKVBlock::HEAD_DIM);
    for (size_t d = 0; d < SparseKVBlock::HEAD_DIM; ++d) {
        query[d] = dist(rng);
    }

    // Initialize blocks with synthetic data
    for (size_t b = 0; b < NUM_BLOCKS; ++b) {
        blocks[b].block_id = static_cast<uint32_t>(b);
        blocks[b].num_tokens = SparseKVBlock::BLOCK_SIZE;
        blocks[b].scale = 0.05f;
        blocks[b].zero_point = -0.075f;
        blocks[b].block_radius = 0.5f;

        // Make block 7 and block 21 highly aligned with query
        for (size_t d = 0; d < SparseKVBlock::HEAD_DIM; ++d) {
            if (b == 7 || b == 21) {
                blocks[b].centroid[d] = query[d] * 2.0f;
            } else {
                blocks[b].centroid[d] = dist(rng) * 0.1f;
            }
        }

        for (size_t i = 0; i < SparseKVBlock::PACKED_BYTES; ++i) {
            blocks[b].packed_keys[i] = static_cast<uint8_t>(rng() & 0xFF);
            blocks[b].packed_values[i] = static_cast<uint8_t>(rng() & 0xFF);
        }
    }

    AttentionSparsityConfig config;
    config.threshold_delta = 0.05f;
    WeftSparseAttentionEngine engine(config);

    std::vector<uint32_t> active_indices(NUM_BLOCKS);
    float max_logit = 0.0f;
    size_t active_count = engine.screen_blocks(query.data(), blocks.data(), NUM_BLOCKS, active_indices.data(), max_logit);

    std::cout << "[TEST] Total Blocks: " << NUM_BLOCKS << ", Active Blocks Selected: " << active_count << std::endl;
    std::cout << "[TEST] Pruning Sparsity: " << (100.0f * (NUM_BLOCKS - active_count) / NUM_BLOCKS) << "%" << std::endl;

    assert(active_count > 0 && active_count <= NUM_BLOCKS);
    // Blocks 7 and 21 should be selected
    bool has_7 = false, has_21 = false;
    for (size_t i = 0; i < active_count; ++i) {
        if (active_indices[i] == 7) has_7 = true;
        if (active_indices[i] == 21) has_21 = true;
    }
    assert(has_7);
    assert(has_21);

    std::vector<float> out_attn(SparseKVBlock::HEAD_DIM);
    engine.compute_sparse_attention(query.data(), blocks.data(), active_indices.data(), active_count, out_attn.data());

    float sum_abs = 0.0f;
    for (size_t d = 0; d < SparseKVBlock::HEAD_DIM; ++d) {
        sum_abs += std::abs(out_attn[d]);
    }
    assert(sum_abs > 0.0f);
    std::cout << "[TEST] Accumulated Sparse Attention Absolute Norm: " << sum_abs << std::endl;

    std::cout << "[TEST] All sparse attention pruning checks passed successfully!" << std::endl;
    return 0;
}
