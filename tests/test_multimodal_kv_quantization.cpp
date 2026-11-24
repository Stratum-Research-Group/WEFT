// test_multimodal_kv_quantization.cpp - Verification for Paper #128
#include "csrc/cli/weft_multimodal_kv.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace weft;

int main() {
    std::cout << "[TEST] Initializing test_multimodal_kv_quantization..." << std::endl;

    std::vector<float> query(MultiModalTokenBlock::DIM, 1.0f); // Query vector of all 1.0s

    // Test 1: Vision Token (Ternary {-1, 0, +1})
    MultiModalTokenBlock vis_block;
    vis_block.modality = ModalityType::VISION;
    vis_block.scale = 0.5f;

    // First 16 coordinates +1, next 16 coordinates -1, remaining 32 coordinates 0
    vis_block.pos_mask = 0x000000000000FFFFULL;
    vis_block.neg_mask = 0x00000000FFFF0000ULL;

    float vis_dot = WeftMultiModalKVEngine::compute_token_dot_product(query.data(), vis_block);
    // (16 * 1.0 - 16 * 1.0) * 0.5 = 0.0
    assert(std::abs(vis_dot) < 1e-5f);
    std::cout << "[TEST] Balanced ternary visual dot product: " << vis_dot << " (Verified 0.0)" << std::endl;

    // Make positive unbalanced: 32 positive (+1), 0 negative
    vis_block.pos_mask = 0x00000000FFFFFFFFULL;
    vis_block.neg_mask = 0x0ULL;
    vis_dot = WeftMultiModalKVEngine::compute_token_dot_product(query.data(), vis_block);
    // 32 * 1.0 * 0.5 / sqrt(64) = 16.0 / 8.0 = 2.0
    assert(std::abs(vis_dot - 2.0f) < 1e-5f);
    std::cout << "[TEST] Unbalanced ternary visual dot product: " << vis_dot << " (Verified 2.0)" << std::endl;

    // Test 2: Text Token (2-bit uniform)
    MultiModalTokenBlock text_block;
    text_block.modality = ModalityType::TEXT;
    text_block.scale = 0.1f;
    text_block.zero_point = 0.0f;
    for (size_t i = 0; i < sizeof(text_block.text_packed); ++i) {
        // All coordinates set to discrete state 2 (2 * 0.1 = 0.2)
        text_block.text_packed[i] = 0b10101010;
    }

    float text_dot = WeftMultiModalKVEngine::compute_token_dot_product(query.data(), text_block);
    // 64 * 1.0 * 0.2 / sqrt(64) = 12.8 / 8.0 = 1.6
    assert(std::abs(text_dot - 1.6f) < 1e-5f);
    std::cout << "[TEST] 2-bit text token dot product: " << text_dot << " (Verified 1.6)" << std::endl;

    std::cout << "[TEST] All multimodal heterogeneous KV quantization checks passed successfully!" << std::endl;
    return 0;
}
