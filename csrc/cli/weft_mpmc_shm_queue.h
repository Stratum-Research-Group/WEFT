#pragma once
// weft_mpmc_shm_queue.h - Lock-Free Multi-Producer Multi-Consumer (MPMC) Queue for Swarm IPC
// Milestone Paper #127
// Bounded lock-free MPMC queue using atomic sequence tickets and cache-line padding (64 bytes).

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace weft {

template <typename T, size_t Capacity = 1024>
class WeftMPMCShmQueue {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");

    struct Node {
        std::atomic<size_t> sequence{0};
        T data;
    };

    struct alignas(64) Header {
        alignas(64) std::atomic<size_t> enqueue_pos{0};
        alignas(64) std::atomic<size_t> dequeue_pos{0};
        size_t capacity{Capacity};
    };

    struct Layout {
        Header header;
        Node buffer[Capacity];
    };

    WeftMPMCShmQueue(const char* shm_name = "/weft_mpmc_queue", bool is_creator = false)
        : shm_name_(shm_name), fd_(-1), layout_(nullptr), is_creator_(is_creator) {
        
        int flags = O_RDWR;
        if (is_creator_) flags |= O_CREAT;
        fd_ = shm_open(shm_name_, flags, 0666);
        if (fd_ >= 0) {
            if (is_creator_) {
                ftruncate(fd_, sizeof(Layout));
            }
            void* addr = mmap(nullptr, sizeof(Layout), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
            if (addr != MAP_FAILED) {
                layout_ = static_cast<Layout*>(addr);
                if (is_creator_) {
                    layout_->header.enqueue_pos.store(0, std::memory_order_relaxed);
                    layout_->header.dequeue_pos.store(0, std::memory_order_relaxed);
                    layout_->header.capacity = Capacity;
                    for (size_t i = 0; i < Capacity; ++i) {
                        layout_->buffer[i].sequence.store(i, std::memory_order_relaxed);
                    }
                }
            }
        }
    }

    ~WeftMPMCShmQueue() {
        if (layout_ && layout_ != MAP_FAILED) {
            munmap(layout_, sizeof(Layout));
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    bool is_valid() const {
        return layout_ != nullptr && layout_ != MAP_FAILED;
    }

    // Concurrent enqueue using atomic ticket fetch-and-add
    bool enqueue(const T& data) {
        if (!is_valid()) return false;
        Node* node;
        size_t pos = layout_->header.enqueue_pos.load(std::memory_order_relaxed);

        for (;;) {
            node = &layout_->buffer[pos & (Capacity - 1)];
            size_t seq = node->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

            if (diff == 0) {
                if (layout_->header.enqueue_pos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false; // Queue full
            } else {
                pos = layout_->header.enqueue_pos.load(std::memory_order_relaxed);
            }
        }

        node->data = data;
        node->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    // Concurrent dequeue using atomic ticket fetch-and-add
    bool dequeue(T& out_data) {
        if (!is_valid()) return false;
        Node* node;
        size_t pos = layout_->header.dequeue_pos.load(std::memory_order_relaxed);

        for (;;) {
            node = &layout_->buffer[pos & (Capacity - 1)];
            size_t seq = node->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);

            if (diff == 0) {
                if (layout_->header.dequeue_pos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false; // Queue empty
            } else {
                pos = layout_->header.dequeue_pos.load(std::memory_order_relaxed);
            }
        }

        out_data = node->data;
        node->sequence.store(pos + Capacity, std::memory_order_release);
        return true;
    }

private:
    const char* shm_name_;
    int fd_;
    Layout* layout_;
    bool is_creator_;
};

} // namespace weft
