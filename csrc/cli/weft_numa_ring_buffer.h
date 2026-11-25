#ifndef WEFT_NUMA_RING_BUFFER_H
#define WEFT_NUMA_RING_BUFFER_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace weft {

// NUMA-Aware Edge Zero-Copy Ring Buffer for Speculative KV Verification
// Single-Producer Single-Consumer (SPSC) lock-free ring buffer aligned to 64-byte cachelines.
template <typename T, size_t Capacity = 1024>
class NUMARingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

public:
    NUMARingBuffer() : head_(0), tail_(0) {
        buffer_.resize(Capacity);
    }

    // Producer pushes speculative token / KV reference
    bool push(const T& item) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t current_head = head_.load(std::memory_order_acquire);

        if ((current_tail - current_head) >= Capacity) {
            return false; // Buffer full
        }

        buffer_[current_tail & (Capacity - 1)] = item;
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    // Consumer (verifier model) pops speculative token
    bool pop(T& item) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        size_t current_tail = tail_.load(std::memory_order_acquire);

        if (current_head == current_tail) {
            return false; // Buffer empty
        }

        item = buffer_[current_head & (Capacity - 1)];
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    size_t size() const {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t tail = tail_.load(std::memory_order_relaxed);
        return (tail >= head) ? (tail - head) : 0;
    }

    bool empty() const {
        return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);
    }

private:
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    std::vector<T> buffer_;
};

} // namespace weft

#endif // WEFT_NUMA_RING_BUFFER_H
