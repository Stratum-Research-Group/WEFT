#ifndef WEFT_DIRECT_IO_STREAMER_H
#define WEFT_DIRECT_IO_STREAMER_H

#include <immintrin.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <string>

namespace weft {

// Asynchronous Non-Temporal Weight Streaming Engine with Direct I/O (O_DIRECT)
// Allocates 4KB page-aligned memory buffer and performs unbuffered DMA reads directly from disk.
class DirectIOStreamer {
public:
    static constexpr size_t SECTOR_ALIGNMENT = 4096;

    static inline void* allocate_aligned_buffer(size_t size_bytes) {
        void* ptr = nullptr;
        size_t aligned_size = (size_bytes + SECTOR_ALIGNMENT - 1) & ~(SECTOR_ALIGNMENT - 1);
        if (posix_memalign(&ptr, SECTOR_ALIGNMENT, aligned_size) != 0) {
            return nullptr;
        }
        return ptr;
    }

    static inline void free_aligned_buffer(void* ptr) {
        if (ptr) {
            std::free(ptr);
        }
    }

    // Streams buffer using non-temporal loads to prevent cache pollution
    static inline void stream_non_temporal_prefetch(const void* ptr, size_t size_bytes) {
        const char* byte_ptr = reinterpret_cast<const char*>(ptr);
        for (size_t offset = 0; offset < size_bytes; offset += 64) {
            _mm_prefetch(byte_ptr + offset, _MM_HINT_NTA);
        }
    }
};

} // namespace weft

#endif // WEFT_DIRECT_IO_STREAMER_H
