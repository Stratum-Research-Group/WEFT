#pragma once
// weft_sliding_attention.h - Dynamic Context Window Compaction & Sliding-Window Attention
// Milestone Paper #126 / Phase 6
// Keeps initial attention sinks (S = 4) + sliding window (W = 4096) in local DDR5/L3 cache,
// while compacting and evicting intermediate KV pages to avoid OOM in continuous agent swarms.

#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <memory>
#include "weft_numa_paged_kv.h"

namespace weft {

class WeftSlidingAttentionCompactor {
public:
    size_t sink_tokens;
    size_t window_size;
    size_t tokens_per_page;

    WeftSlidingAttentionCompactor(size_t sink_tokens = 4, size_t window_size = 4096, size_t tokens_per_page = 16)
        : sink_tokens(sink_tokens), window_size(window_size), tokens_per_page(tokens_per_page) {}

    // Determines if a page should be preserved in hot local tier or compacted/evicted
    bool is_page_in_active_window(size_t page_id, size_t total_tokens) const {
        if (total_tokens <= window_size + sink_tokens) {
            return true; // Whole context fits in window
        }

        size_t page_start_token = page_id * tokens_per_page;
        size_t page_end_token = page_start_token + tokens_per_page;

        // 1. Attention Sink: Always preserve initial tokens
        if (page_start_token < sink_tokens) {
            return true;
        }

        // 2. Sliding Window: Preserve tokens in [total_tokens - window_size, total_tokens]
        size_t window_start_token = total_tokens - window_size;
        if (page_end_token > window_start_token) {
            return true;
        }

        // 3. Intermediate tokens are eligible for compaction/remote migration
        return false;
    }

    // Compacts a NUMA KV pool by demoting out-of-window pages to cold remote tier
    size_t compact_pool(WeftNUMAPagedKVPool& pool, size_t active_pages, size_t total_tokens, uint8_t remote_node = 1) {
        size_t demoted_count = 0;
        for (size_t p = 0; p < active_pages; ++p) {
            if (!is_page_in_active_window(p, total_tokens)) {
                NUMAPageDescriptor* desc = pool.get_page(p);
                if (desc && desc->tier == MemoryTier::HOT_LOCAL) {
                    desc->tier = MemoryTier::COLD_REMOTE;
                    desc->numa_node = remote_node;
                    desc->access_counter = 0;
                    demoted_count++;
                }
            }
        }
        return demoted_count;
    }
};

} // namespace weft
