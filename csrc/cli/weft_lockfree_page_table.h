#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <atomic>
#include "weft_paged_kv.h"

namespace weft {

constexpr uint32_t WEFT_NULL_PAGE = 0xFFFFFFFF;

class WeftLockFreePagePool {
public:
    size_t total_pages;
    alignas(64) std::atomic<uint64_t> head{0};
    std::vector<uint32_t> next_free;
    std::vector<WeftKVBlock> page_storage;

    WeftLockFreePagePool(size_t pool_size = 1024, size_t dim = 64)
        : total_pages(pool_size), next_free(pool_size), page_storage(pool_size) {
        // Initialize lock-free freelist chain
        for (uint32_t i = 0; i < pool_size - 1; ++i) {
            next_free[i] = i + 1;
        }
        next_free[pool_size - 1] = WEFT_NULL_PAGE;

        // Head points to page 0 with version 0
        head.store(0, std::memory_order_relaxed);

        // Pre-allocate blocks
        for (auto& b : page_storage) {
            b.scale = 1.0f;
            b.zero_point = 0.0f;
            b.num_tokens = 0;
            b.flags = 0;
            b.packed_keys.resize((WEFT_KV_PAGE_TOKENS * dim + 3) / 4, 0);
            b.packed_values.resize((WEFT_KV_PAGE_TOKENS * dim + 3) / 4, 0);
        }
    }

    // Atomic CAS allocation of a free page block
    bool allocate_page(uint32_t& allocated_page_id) {
        uint64_t old_head = head.load(std::memory_order_relaxed);
        uint64_t new_head;
        do {
            uint32_t page_id = static_cast<uint32_t>(old_head & 0xFFFFFFFF);
            if (page_id == WEFT_NULL_PAGE) {
                return false; // Out of memory
            }
            uint32_t next_id = next_free[page_id];
            uint32_t version = static_cast<uint32_t>(old_head >> 32) + 1;
            new_head = (static_cast<uint64_t>(version) << 32) | next_id;
        } while (!head.compare_exchange_weak(old_head, new_head,
                                              std::memory_order_acquire,
                                              std::memory_order_relaxed));

        allocated_page_id = static_cast<uint32_t>(old_head & 0xFFFFFFFF);
        return true;
    }

    // Atomic CAS reclamation of a page block
    void free_page(uint32_t page_id) {
        uint64_t old_head = head.load(std::memory_order_relaxed);
        uint64_t new_head;
        do {
            uint32_t next_id = static_cast<uint32_t>(old_head & 0xFFFFFFFF);
            next_free[page_id] = next_id;
            uint32_t version = static_cast<uint32_t>(old_head >> 32) + 1;
            new_head = (static_cast<uint64_t>(version) << 32) | page_id;
        } while (!head.compare_exchange_weak(old_head, new_head,
                                              std::memory_order_release,
                                              std::memory_order_relaxed));
    }
};

} // namespace weft
