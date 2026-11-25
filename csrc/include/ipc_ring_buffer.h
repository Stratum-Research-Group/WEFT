#pragma once
#include <cstdint>
#include <atomic>
#include <cstring>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace weft {

constexpr size_t RING_CAPACITY = 1024;

struct alignas(64) TokenMessage {
    int64_t seq_id;
    int32_t token_id;
    float logprob;
    bool is_final;
};

struct IPCRingBuffer {
    std::atomic<uint64_t> head{0};
    std::atomic<uint64_t> tail{0};
    TokenMessage slots[RING_CAPACITY];

    bool push(const TokenMessage& msg) {
        uint64_t current_tail = tail.load(std::memory_order_relaxed);
        uint64_t current_head = head.load(std::memory_order_acquire);
        if (current_tail - current_head >= RING_CAPACITY) {
            return false; // Queue full
        }
        slots[current_tail % RING_CAPACITY] = msg;
        tail.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    bool pop(TokenMessage& out) {
        uint64_t current_head = head.load(std::memory_order_relaxed);
        uint64_t current_tail = tail.load(std::memory_order_acquire);
        if (current_head == current_tail) {
            return false; // Queue empty
        }
        out = slots[current_head % RING_CAPACITY];
        head.store(current_head + 1, std::memory_order_release);
        return true;
    }
};

} // namespace weft
