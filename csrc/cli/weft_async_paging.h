#pragma once
// weft_async_paging.h - Direct Kernel-Level Async io_uring Page Streaming
// Milestone Paper #126 / Phase 6
// Integrates Linux io_uring direct NVMe/CXL DMA streaming into NUMA KV cache pool.

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include <cstring>
#include <atomic>
#include <iostream>
#include "../loader/weft_iouring_loader.h"
#include "weft_numa_paged_kv.h"

namespace weft {

class WeftAsyncPageStreamer {
public:
    WeftAsyncPageStreamer(WeftNUMAPagedKVPool& pool, size_t queue_depth = 64)
        : pool_(pool), loader_(queue_depth), pending_requests_(0) {}

    // Initialize backing store descriptor
    bool open_backing_store(const char* filepath) {
        return loader_.open_file(filepath);
    }

    // Prefetch cold page from backing NVMe store directly into local NUMA page buffer
    bool prefetch_page_async(uint32_t page_id, uint64_t file_offset) {
        NUMAPageDescriptor* desc = pool_.get_page(page_id);
        if (!desc) return false;

        // Queue asynchronous zero-syscall read request via io_uring
        bool submitted = loader_.submit_read(file_offset, NUMAPageDescriptor::PAGE_SIZE_BYTES, desc->data);
        if (submitted) {
            pending_requests_.fetch_add(1, std::memory_order_relaxed);
            desc->tier = MemoryTier::HOT_LOCAL;
            desc->access_counter = 128; // Warm reset
        }
        return submitted;
    }

    // Reap completed DMA transfers
    size_t poll_completions(size_t min_complete = 1) {
        size_t completed = loader_.reap_completions(min_complete);
        pending_requests_.fetch_sub(completed, std::memory_order_relaxed);
        return completed;
    }

    size_t pending_requests() const {
        return pending_requests_.load(std::memory_order_relaxed);
    }

private:
    WeftNUMAPagedKVPool& pool_;
    WeftIOUringLoader loader_;
    std::atomic<size_t> pending_requests_;
};

} // namespace weft
