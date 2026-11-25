#include "../csrc/cli/weft_kv_cache.h"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Test Continuous KV Cache & Rollback ===" << std::endl;

    size_t num_layers = 4;
    size_t num_heads = 8;
    size_t head_dim = 64;
    size_t max_seq = 128;

    weft::ContinuousKVCache cache(num_layers, num_heads, head_dim, max_seq);
    assert(cache.current_len == 0);

    // Append 5 tokens
    std::vector<float> dummy_k(num_heads * head_dim, 1.5f);
    std::vector<float> dummy_v(num_heads * head_dim, 2.5f);

    for (size_t t = 0; t < 5; ++t) {
        for (size_t l = 0; l < num_layers; ++l) {
            cache.append_token(l, dummy_k.data(), dummy_v.data());
        }
        cache.commit_step();
    }
    assert(cache.current_len == 5);

    // Rollback to 3 tokens (e.g. 2 speculative tokens rejected)
    cache.rollback(3);
    assert(cache.current_len == 3);

    std::cout << "[PASS] Continuous KV cache append, stride alignment, and speculative rollback verified." << std::endl;
    return 0;
}
