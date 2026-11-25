#ifndef WEFT_SHM_BROADCAST_BUS_H
#define WEFT_SHM_BROADCAST_BUS_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace weft {

// Multi-Node NUMA-Aware Inter-Process Shared Memory Token Broadcast Bus
// Lock-free circular token broadcast structure designed for mmap(MAP_SHARED) execution.
struct alignas(64) SHMBroadcastSlot {
    std::atomic<uint64_t> sequence{0};
    uint32_t token_id{0};
    float token_prob{0.0f};
    uint32_t model_source_id{0};
};

template <size_t Capacity = 1024>
class SHMBroadcastBus {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of two");

public:
    SHMBroadcastBus() : write_seq_(0) {
        std::memset(slots_, 0, sizeof(slots_));
    }

    // Broadcast a new token from publisher model
    void broadcast(uint32_t token_id, float prob, uint32_t model_id) {
        uint64_t seq = write_seq_.fetch_add(1, std::memory_order_relaxed);
        size_t idx = seq & (Capacity - 1);

        slots_[idx].token_id = token_id;
        slots_[idx].token_prob = prob;
        slots_[idx].model_source_id = model_id;
        slots_[idx].sequence.store(seq + 1, std::memory_order_release);
    }

    // Read the latest token if available
    bool read_latest(uint64_t& last_read_seq, uint32_t& out_token, float& out_prob, uint32_t& out_model) const {
        uint64_t current_seq = write_seq_.load(std::memory_order_relaxed);
        if (current_seq == 0 || current_seq <= last_read_seq) {
            return false; // No new tokens
        }

        uint64_t target_seq = current_seq - 1;
        size_t idx = target_seq & (Capacity - 1);

        // Wait until publication completes
        if (slots_[idx].sequence.load(std::memory_order_acquire) != target_seq + 1) {
            return false;
        }

        out_token = slots_[idx].token_id;
        out_prob = slots_[idx].token_prob;
        out_model = slots_[idx].model_source_id;
        last_read_seq = current_seq;
        return true;
    }

    uint64_t current_sequence() const {
        return write_seq_.load(std::memory_order_relaxed);
    }

private:
    alignas(64) std::atomic<uint64_t> write_seq_;
    alignas(64) SHMBroadcastSlot slots_[Capacity];
};

} // namespace weft

#endif // WEFT_SHM_BROADCAST_BUS_H
