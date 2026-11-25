// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0
//
// Multi-Threaded NUMA-Aware Edge Token Decoding Streamer (Item 5)
// Lock-free ring buffer for streaming generated token IDs to HTTP/SSE workers.

#ifndef WEFT_NUMA_TOKEN_STREAMER_H
#define WEFT_NUMA_TOKEN_STREAMER_H

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <vector>

namespace weft {

class NumaTokenStreamer {
private:
    static constexpr size_t BUFFER_SIZE = 1024;
    uint32_t ring_buffer_[BUFFER_SIZE];
    std::atomic<size_t> write_head_{0};
    std::atomic<size_t> read_head_{0};

public:
    NumaTokenStreamer() {
        for (size_t i = 0; i < BUFFER_SIZE; ++i) {
            ring_buffer_[i] = 0;
        }
    }

    bool push_token(uint32_t token_id) {
        size_t current_w = write_head_.load(std::memory_order_relaxed);
        size_t current_r = read_head_.load(std::memory_order_acquire);
        if (current_w - current_r >= BUFFER_SIZE) {
            return false; // buffer full
        }
        ring_buffer_[current_w % BUFFER_SIZE] = token_id;
        write_head_.store(current_w + 1, std::memory_order_release);
        return true;
    }

    bool pop_token(uint32_t& out_token) {
        size_t current_r = read_head_.load(std::memory_order_relaxed);
        size_t current_w = write_head_.load(std::memory_order_acquire);
        if (current_r >= current_w) {
            return false; // buffer empty
        }
        out_token = ring_buffer_[current_r % BUFFER_SIZE];
        read_head_.store(current_r + 1, std::memory_order_release);
        return true;
    }

    size_t pending_count() const {
        size_t w = write_head_.load(std::memory_order_relaxed);
        size_t r = read_head_.load(std::memory_order_relaxed);
        return (w >= r) ? (w - r) : 0;
    }
};

} // namespace weft

#endif // WEFT_NUMA_TOKEN_STREAMER_H
