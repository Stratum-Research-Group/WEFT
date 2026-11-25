#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace weft {

// Lock-Free Multi-Tenant Shared KV Memory Allocator for Edge Serving
// Provides fixed-size KV page chunk allocation using atomic free-list stack
class SharedKVAllocator {
public:
    struct PageChunk {
        uint32_t tenant_id;
        uint32_t page_index;
        uint32_t next_free;
    };

    SharedKVAllocator(size_t total_pages, size_t page_size_bytes = 4096)
        : total_pages_(total_pages), page_size_(page_size_bytes), head_free_(0) {
        pool_.resize(total_pages * page_size_, 0);
        meta_.resize(total_pages);
        for (size_t i = 0; i < total_pages; ++i) {
            meta_[i].tenant_id = 0;
            meta_[i].page_index = static_cast<uint32_t>(i);
            meta_[i].next_free = static_cast<uint32_t>(i + 1);
        }
        meta_[total_pages - 1].next_free = NULL_INDEX;
    }

    int32_t allocate_page(uint32_t tenant_id) {
        uint32_t curr = head_free_.load(std::memory_order_relaxed);
        while (curr != NULL_INDEX) {
            uint32_t next = meta_[curr].next_free;
            if (head_free_.compare_exchange_weak(curr, next, std::memory_order_acquire, std::memory_order_relaxed)) {
                meta_[curr].tenant_id = tenant_id;
                meta_[curr].next_free = NULL_INDEX;
                return static_cast<int32_t>(curr);
            }
        }
        return -1; // Out of memory
    }

    bool free_page(int32_t page_id, uint32_t tenant_id) {
        if (page_id < 0 || static_cast<size_t>(page_id) >= total_pages_) return false;
        if (meta_[page_id].tenant_id != tenant_id) return false; // Security boundary

        meta_[page_id].tenant_id = 0;
        uint32_t curr = head_free_.load(std::memory_order_relaxed);
        do {
            meta_[page_id].next_free = curr;
        } while (!head_free_.compare_exchange_weak(curr, static_cast<uint32_t>(page_id),
                                                   std::memory_order_release, std::memory_order_relaxed));
        return true;
    }

    uint8_t* get_page_buffer(int32_t page_id, uint32_t tenant_id) {
        if (page_id < 0 || static_cast<size_t>(page_id) >= total_pages_) return nullptr;
        if (meta_[page_id].tenant_id != tenant_id) return nullptr; // Tenant isolation
        return &pool_[page_id * page_size_];
    }

    size_t available_pages() const {
        size_t count = 0;
        uint32_t curr = head_free_.load(std::memory_order_relaxed);
        while (curr != NULL_INDEX && count <= total_pages_) {
            count++;
            curr = meta_[curr].next_free;
        }
        return count;
    }

private:
    static constexpr uint32_t NULL_INDEX = 0xFFFFFFFF;
    size_t total_pages_;
    size_t page_size_;
    std::atomic<uint32_t> head_free_;
    std::vector<PageChunk> meta_;
    std::vector<uint8_t> pool_;
};

} // namespace weft
