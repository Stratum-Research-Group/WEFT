#include "csrc/cli/weft_kv_prefetcher.h"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    constexpr size_t total_tokens = 256;
    constexpr size_t kv_dim = 128;
    std::vector<float> kv_buffer(total_tokens * kv_dim, 1.0f);

    // Call prefetcher across multiple positions
    for (size_t pos = 0; pos < 200; pos += 16) {
        weft::TokenKVPrefetcher::prefetch_token_kv_lines(kv_buffer.data(), pos, 4, kv_dim);
        weft::TokenKVPrefetcher::prefetch_next_page(kv_buffer.data(), pos * kv_dim * sizeof(float));
    }

    // Check buffer remains valid and accessible
    assert(kv_buffer[0] == 1.0f);
    assert(kv_buffer[total_tokens * kv_dim - 1] == 1.0f);

    std::cout << "[PASS] Token KV Prefetcher Unit Test passed.\n";
    return 0;
}
