#pragma once
// weft_numa_paged_kv.h - Asymmetric Multi-Tier KV Cache Paging & NUMA Memory Pool
// Milestone Paper #126
// Authors: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstdint>
#include <cstddef>
#include <vector>
#include <atomic>
#include <memory>
#include <cstring>
#include <cassert>

namespace weft {

enum class MemoryTier : uint8_t {
    HOT_LOCAL = 0,    // Local NUMA node DDR5 / L3 cache
    COLD_REMOTE = 1   // Remote NUMA node / CXL memory pool
};

struct NUMAPageDescriptor {
    static constexpr size_t PAGE_SIZE_BYTES = 4096; // 4 KB page
    static constexpr size_t TOKENS_PER_PAGE = 16;
    static constexpr size_t HEAD_DIM = 64;

    uint32_t page_id{0};
    uint8_t numa_node{0};
    MemoryTier tier{MemoryTier::HOT_LOCAL};
    uint8_t access_counter{0}; // 8-bit exponential decay hotness counter

    alignas(64) uint8_t data[PAGE_SIZE_BYTES];
};

class WeftNUMAPagedKVPool {
public:
    explicit WeftNUMAPagedKVPool(size_t total_pages = 256, uint8_t local_node = 0)
        : total_pages_(total_pages), local_node_(local_node) {
        pages_.resize(total_pages_);
        for (size_t i = 0; i < total_pages_; ++i) {
            pages_[i].page_id = static_cast<uint32_t>(i);
            pages_[i].numa_node = local_node_;
            pages_[i].tier = MemoryTier::HOT_LOCAL;
            pages_[i].access_counter = 0;
            std::memset(pages_[i].data, 0, NUMAPageDescriptor::PAGE_SIZE_BYTES);
        }
    }

    // Access page and record access heat
    NUMAPageDescriptor* get_page(size_t page_id) {
        if (page_id >= total_pages_) return nullptr;
        auto& page = pages_[page_id];
        // Saturating counter increment
        if (page.access_counter < 255) {
            page.access_counter++;
        }
        return &page;
    }

    // Migrate a cold page from remote to local NUMA tier
    bool migrate_page_to_local(size_t page_id) {
        if (page_id >= total_pages_) return false;
        auto& page = pages_[page_id];
        if (page.tier == MemoryTier::HOT_LOCAL && page.numa_node == local_node_) {
            return true; // Already local
        }

        // Simulate zero-copy / DMA migration
        page.numa_node = local_node_;
        page.tier = MemoryTier::HOT_LOCAL;
        page.access_counter = 128; // Reset to warm baseline
        return true;
    }

    // Demote cold pages with access_counter below threshold to cold remote tier
    size_t evict_cold_pages(uint8_t cold_threshold, uint8_t remote_node) {
        size_t demoted = 0;
        for (auto& page : pages_) {
            if (page.access_counter < cold_threshold && page.tier == MemoryTier::HOT_LOCAL) {
                page.tier = MemoryTier::COLD_REMOTE;
                page.numa_node = remote_node;
                demoted++;
            }
        }
        return demoted;
    }

    // Count pages by tier
    void get_tier_counts(size_t& out_hot, size_t& out_cold) const {
        out_hot = 0;
        out_cold = 0;
        for (const auto& page : pages_) {
            if (page.tier == MemoryTier::HOT_LOCAL) out_hot++;
            else out_cold++;
        }
    }

private:
    size_t total_pages_;
    uint8_t local_node_;
    std::vector<NUMAPageDescriptor> pages_;
};

} // namespace weft
