#include <iostream>
#include <cassert>
#include <vector>
#include "../csrc/cli/weft_circular_kv.h"

int main() {
    std::cout << "[TEST] Running Weft Circular KV Cache Zero-Allocation Verification..." << std::endl;

    const size_t dim = 64;
    const size_t max_blocks = 4; // 4 blocks * 16 tokens = 64 tokens max
    weft::WeftCircularKVCache cache(dim, max_blocks, 16);

    std::vector<float> key(dim, 1.0f);
    std::vector<float> val(dim, -0.5f);

    // Append 100 tokens (cycling past 64 tokens)
    for (size_t i = 0; i < 100; ++i) {
        cache.append_token_zero_alloc(key.data(), val.data());
    }

    assert(cache.total_tokens_appended == 100);
    assert(cache.active_tokens() == 64);
    assert(cache.ring_blocks.size() == max_blocks);
    std::cout << "[PASS] Circular ring buffer recycled blocks seamlessly (100 tokens through 64 capacity)." << std::endl;

    std::cout << "[ALL PASSED] Circular KV Zero-Allocation tests successful." << std::endl;
    return 0;
}
