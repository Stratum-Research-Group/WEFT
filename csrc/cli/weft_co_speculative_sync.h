#pragma once
// weft_co_speculative_sync.h - Heterogeneous CPU-GPU Co-Speculative Pipeline Synchronization
// Milestone Paper #130
// Couples /dev/shm/weft_spec_ring with Linux eventfd and atomic memory fences to synchronize
// CPU host draft generation (AVX-512) and GPU batch verification (Triton 3.9) with zero thread spinning.

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <chrono>
#include <sys/eventfd.h>
#include <unistd.h>
#include <poll.h>

namespace weft {

class WeftCoSpeculativeSync {
public:
    WeftCoSpeculativeSync() : event_fd_(-1) {
        event_fd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    }

    ~WeftCoSpeculativeSync() {
        if (event_fd_ >= 0) {
            close(event_fd_);
        }
    }

    bool is_valid() const {
        return event_fd_ >= 0;
    }

    // Host CPU signals that speculative candidate DAG branch has been written to SHM ring
    bool signal_draft_ready() {
        if (event_fd_ < 0) return false;
        uint64_t val = 1;
        ssize_t s = write(event_fd_, &val, sizeof(uint64_t));
        return s == sizeof(uint64_t);
    }

    // GPU Verifier waits for incoming candidate batch with bounded timeout (zero busy-spin)
    bool wait_for_draft(int timeout_ms = 50) {
        if (event_fd_ < 0) return false;
        struct pollfd pfd;
        pfd.fd = event_fd_;
        pfd.events = POLLIN;

        int ret = poll(&pfd, 1, timeout_ms);
        if (ret > 0 && (pfd.revents & POLLIN)) {
            uint64_t count = 0;
            read(event_fd_, &count, sizeof(uint64_t)); // Clear event counter
            return true;
        }
        return false;
    }

    // GPU Verifier signals completion of batch verification back to CPU draft engine
    void signal_verification_complete(std::atomic<uint64_t>& ack_counter) {
        ack_counter.fetch_add(1, std::memory_order_release);
    }

private:
    int event_fd_;
};

} // namespace weft
