// test_entropy_kv_compaction.cpp - Verification for Paper #136
#include "csrc/cli/weft_entropy_kv.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace weft;

int main() {
    std::cout << "[TEST] Initializing test_entropy_kv_compaction..." << std::endl;

    WeftEntropyKVEngine engine(/*tau_high=*/1.8f, /*tau_low=*/0.5f);

    // Test 1: Predictable Token (one dominant logit -> low entropy < 0.5)
    float low_entropy_logits[8] = {10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float h_low = WeftEntropyKVEngine::compute_entropy_surrogate(low_entropy_logits, 8);
    std::cout << "[TEST] Predictable Logits Entropy: " << h_low << " bits" << std::endl;
    assert(h_low < 0.5f);
    assert(engine.classify_tier(h_low) == EntropyPrecisionTier::LOW_TERNARY);

    // Test 2: High Entropy Token (uniform distribution -> maximum entropy = log2(8) = 3.0 bits)
    float high_entropy_logits[8] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    float h_high = WeftEntropyKVEngine::compute_entropy_surrogate(high_entropy_logits, 8);
    std::cout << "[TEST] Uniform Branching Entropy: " << h_high << " bits (Expected ~3.0)" << std::endl;
    assert(std::abs(h_high - 3.0f) < 1e-4f);
    assert(engine.classify_tier(h_high) == EntropyPrecisionTier::HIGH_4BIT);

    // Test 3: Moderate Context Token (entropy ~1.0 bit)
    float med_entropy_logits[8] = {3.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float h_med = WeftEntropyKVEngine::compute_entropy_surrogate(med_entropy_logits, 8);
    std::cout << "[TEST] Bimodal Logits Entropy: " << h_med << " bits (Expected ~1.0)" << std::endl;
    assert(h_med >= 0.5f && h_med < 1.8f);
    assert(engine.classify_tier(h_med) == EntropyPrecisionTier::MED_2BIT);

    // Test 4: Verify Dot Product Across Tiers
    std::vector<float> query(EntropyKVBlock::DIM, 1.0f);
    EntropyKVBlock block_4bit;
    block_4bit.tier = EntropyPrecisionTier::HIGH_4BIT;
    block_4bit.scale = 0.1f;
    block_4bit.zero_point = 0.0f;
    for (size_t i = 0; i < 32; ++i) {
        // Nibbles set to 5: 0x55
        block_4bit.packed_data[i] = 0x55;
    }
    float dot_4bit = WeftEntropyKVEngine::compute_token_dot_product(query.data(), block_4bit);
    // 64 * 1.0 * (5 * 0.1) / sqrt(64) = 64 * 0.5 / 8 = 32.0 / 8 = 4.0
    std::cout << "[TEST] 4-bit Token Dot Product: " << dot_4bit << " (Expected 4.0)" << std::endl;
    assert(std::abs(dot_4bit - 4.0f) < 1e-5f);

    std::cout << "[TEST] All entropy-guided dynamic KV compaction checks passed successfully!" << std::endl;
    return 0;
}
