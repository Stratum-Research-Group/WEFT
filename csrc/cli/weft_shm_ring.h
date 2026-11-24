#pragma once
#include <cstdint>
#include <cstddef>
#include <atomic>
#include <array>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>

namespace weft {

constexpr size_t WEFT_SHM_RING_CAPACITY = 256;
constexpr const char* WEFT_SHM_PATH = "/weft_spec_ring_shm";

struct alignas(64) WeftShmSlot {
    std::atomic<uint32_t> sequence_id;
    int32_t token_id;
    float confidence;
    uint32_t flags; // 0x1: draft, 0x2: verified, 0x4: accepted
};

struct alignas(64) WeftShmHeader {
    std::atomic<size_t> head{0};
    std::atomic<size_t> tail{0};
    uint32_t capacity{WEFT_SHM_RING_CAPACITY};
    uint32_t version{1};
    std::array<WeftShmSlot, WEFT_SHM_RING_CAPACITY> slots;
};

class WeftShmRingBuffer {
private:
    int shm_fd{-1};
    WeftShmHeader* header{nullptr};
    bool is_owner{false};

public:
    WeftShmRingBuffer(bool create_as_owner = false) : is_owner(create_as_owner) {
        int flags = O_RDWR;
        if (create_as_owner) {
            flags |= O_CREAT;
        }

        shm_fd = shm_open(WEFT_SHM_PATH, flags, 0666);
        if (shm_fd < 0) {
            // Fallback to anonymous local mapping if POSIX shm is restricted
            header = new WeftShmHeader();
            return;
        }

        if (create_as_owner) {
            if (ftruncate(shm_fd, sizeof(WeftShmHeader)) != 0) {
                close(shm_fd);
                header = new WeftShmHeader();
                return;
            }
        }

        void* ptr = mmap(nullptr, sizeof(WeftShmHeader), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (ptr == MAP_FAILED) {
            close(shm_fd);
            header = new WeftShmHeader();
            return;
        }

        header = static_cast<WeftShmHeader*>(ptr);
        if (create_as_owner) {
            header->head.store(0, std::memory_order_relaxed);
            header->tail.store(0, std::memory_order_relaxed);
            for (size_t i = 0; i < WEFT_SHM_RING_CAPACITY; ++i) {
                header->slots[i].sequence_id.store(0, std::memory_order_relaxed);
                header->slots[i].token_id = -1;
                header->slots[i].confidence = 0.0f;
                header->slots[i].flags = 0;
            }
        }
    }

    ~WeftShmRingBuffer() {
        if (shm_fd >= 0 && header != nullptr) {
            munmap(header, sizeof(WeftShmHeader));
            close(shm_fd);
            if (is_owner) {
                shm_unlink(WEFT_SHM_PATH);
            }
        } else if (header != nullptr) {
            delete header;
        }
    }

    bool enqueue_draft(int32_t token, float conf) {
        size_t current_tail = header->tail.load(std::memory_order_relaxed);
        size_t current_head = header->head.load(std::memory_order_acquire);
        if (current_tail - current_head >= WEFT_SHM_RING_CAPACITY) {
            return false;
        }
        size_t idx = current_tail % WEFT_SHM_RING_CAPACITY;
        header->slots[idx].token_id = token;
        header->slots[idx].confidence = conf;
        header->slots[idx].flags = 0x1;
        header->slots[idx].sequence_id.store(static_cast<uint32_t>(current_tail + 1), std::memory_order_release);
        header->tail.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    bool dequeue_verify(int32_t& token, float& conf) {
        size_t current_head = header->head.load(std::memory_order_relaxed);
        size_t current_tail = header->tail.load(std::memory_order_acquire);
        if (current_head == current_tail) {
            return false;
        }
        size_t idx = current_head % WEFT_SHM_RING_CAPACITY;
        token = header->slots[idx].token_id;
        conf = header->slots[idx].confidence;
        header->slots[idx].flags |= 0x2;
        header->head.store(current_head + 1, std::memory_order_release);
        return true;
    }
};

} // namespace weft
