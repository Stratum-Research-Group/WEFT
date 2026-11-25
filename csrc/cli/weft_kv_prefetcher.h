#ifndef WEFT_KV_PREFETCHER_H
#define WEFT_KV_PREFETCHER_H

#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace weft {

// Real-Time Token KV Cache Prefetcher with OS Virtual Page Paging
// Proactively prefetches token key-value cachelines into L1/L2/L3 cache hierarchies
// across 4KB OS virtual page boundaries during autoregressive decoding.
class TokenKVPrefetcher {
public:
    static constexpr size_t CACHELINE_BYTES = 64;
    static constexpr size_t PAGE_SIZE_BYTES = 4096;

    // Issues hardware prefetch instructions for upcoming tokens
    static inline void prefetch_token_kv_lines(
        const float* __restrict__ kv_buffer,
        size_t current_pos,
        size_t lookahead_tokens,
        size_t kv_dim)
    {
        if (!kv_buffer) return;

        const char* base_ptr = reinterpret_cast<const char*>(kv_buffer);
        size_t token_bytes = kv_dim * sizeof(float);

        for (size_t step = 1; step <= lookahead_tokens; ++step) {
            size_t target_token = current_pos + step;
            const char* token_ptr = base_ptr + target_token * token_bytes;

            // Prefetch every 64-byte cacheline across the token's KV projection
            for (size_t offset = 0; offset < token_bytes; offset += CACHELINE_BYTES) {
                // _MM_HINT_T0: Prefetch into all cache levels (L1/L2/L3)
                _mm_prefetch(token_ptr + offset, _MM_HINT_T0);
            }
        }
    }

    // Prefetches the next virtual memory page ahead of time to avoid TLB miss latency
    static inline void prefetch_next_page(
        const float* __restrict__ kv_buffer,
        size_t current_byte_offset)
    {
        if (!kv_buffer) return;
        const char* base_ptr = reinterpret_cast<const char*>(kv_buffer);
        const char* next_page_ptr = base_ptr + ((current_byte_offset + PAGE_SIZE_BYTES) & ~(PAGE_SIZE_BYTES - 1));
        _mm_prefetch(next_page_ptr, _MM_HINT_T1);
    }
};

} // namespace weft

#endif // WEFT_KV_PREFETCHER_H
