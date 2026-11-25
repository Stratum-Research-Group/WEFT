#pragma once
// weft_shm_channel.h - Zero-Copy Shared Memory Inter-Process Communication Driver
// Low-overhead SPSC atomic lock-free ring channel with cache-line padding (64 bytes).
// Used by external C/C++ tools, IDE sidecars, and runtime engines to communicate with /dev/shm.

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
class WeftShmChannel {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");

    struct alignas(64) Header {
        alignas(64) std::atomic<uint64_t> head{0};
        alignas(64) std::atomic<uint64_t> tail{0};
        uint64_t capacity{Capacity};
        uint64_t item_size{sizeof(T)};
    };

    struct Layout {
        Header header;
        T buffer[Capacity];
    };

    WeftShmChannel(const char* shm_name = "/weft_ipc_channel", bool is_creator = false)
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
                    layout_->header.head.store(0, std::memory_order_relaxed);
                    layout_->header.tail.store(0, std::memory_order_relaxed);
                    layout_->header.capacity = Capacity;
                    layout_->header.item_size = sizeof(T);
                }
            }
        }
    }

    ~WeftShmChannel() {
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

    // Producer: enqueue item into ring buffer
    bool push(const T& item) {
        if (!is_valid()) return false;
        uint64_t tail = layout_->header.tail.load(std::memory_order_relaxed);
        uint64_t head = layout_->header.head.load(std::memory_order_acquire);

        if ((tail - head) >= Capacity) {
            return false; // Queue full
        }

        layout_->buffer[tail & (Capacity - 1)] = item;
        layout_->header.tail.store(tail + 1, std::memory_order_release);
        return true;
    }

    // Consumer: dequeue item from ring buffer
    bool pop(T& out_item) {
        if (!is_valid()) return false;
        uint64_t head = layout_->header.head.load(std::memory_order_relaxed);
        uint64_t tail = layout_->header.tail.load(std::memory_order_acquire);

        if (head >= tail) {
            return false; // Queue empty
        }

        out_item = layout_->buffer[head & (Capacity - 1)];
        layout_->header.head.store(head + 1, std::memory_order_release);
        return true;
    }

    size_t size() const {
        if (!is_valid()) return 0;
        uint64_t head = layout_->header.head.load(std::memory_order_relaxed);
        uint64_t tail = layout_->header.tail.load(std::memory_order_relaxed);
        return (tail > head) ? static_cast<size_t>(tail - head) : 0;
    }

private:
    const char* shm_name_;
    int fd_;
    Layout* layout_;
    bool is_creator_;
};

} // namespace weft
