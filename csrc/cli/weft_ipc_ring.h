#pragma once
#include <cstdint>
#include <cstddef>
#include <atomic>
#include <array>
#include <cstring>

namespace weft {

constexpr size_t WEFT_RING_CAPACITY = 256;

struct alignas(64) WeftRingSlot {
    std::atomic<uint32_t> sequence_id;
    int32_t token_id;
    float confidence;
    uint32_t flags; // 0x1: draft, 0x2: verified, 0x4: accepted
};

class WeftIPCRingBuffer {
public:
    alignas(64) std::atomic<size_t> head{0};
    alignas(64) std::atomic<size_t> tail{0};
    alignas(64) std::array<WeftRingSlot, WEFT_RING_CAPACITY> slots;

    WeftIPCRingBuffer() {
        for (size_t i = 0; i < WEFT_RING_CAPACITY; ++i) {
            slots[i].sequence_id.store(0, std::memory_order_relaxed);
            slots[i].token_id = -1;
            slots[i].confidence = 0.0f;
            slots[i].flags = 0;
        }
    }

    // Producer (Draft Model on GPU/NPU)
    bool enqueue_draft(int32_t token, float conf) {
        size_t current_tail = tail.load(std::memory_order_relaxed);
        size_t current_head = head.load(std::memory_order_acquire);
        if (current_tail - current_head >= WEFT_RING_CAPACITY) {
            return false; // Ring full
        }
        size_t idx = current_tail % WEFT_RING_CAPACITY;
        slots[idx].token_id = token;
        slots[idx].confidence = conf;
        slots[idx].flags = 0x1;
        slots[idx].sequence_id.store(static_cast<uint32_t>(current_tail + 1), std::memory_order_release);
        tail.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    // Consumer (Target Verifier on CPU AVX-512)
    bool dequeue_verify(int32_t& token, float& conf) {
        size_t current_head = head.load(std::memory_order_relaxed);
        size_t current_tail = tail.load(std::memory_order_acquire);
        if (current_head == current_tail) {
            return false; // Empty ring
        }
        size_t idx = current_head % WEFT_RING_CAPACITY;
        token = slots[idx].token_id;
        conf = slots[idx].confidence;
        slots[idx].flags |= 0x2;
        head.store(current_head + 1, std::memory_order_release);
        return true;
    }
};

} // namespace weft
