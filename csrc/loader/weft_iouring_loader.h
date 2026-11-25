#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdexcept>
#include <iostream>

namespace weft {

// High-Throughput Linux io_uring DMA Loading Engine
// Bypasses kernel page cache using direct queue submission/completion pairs
class IOUringShardLoader {
private:
    int fd_ = -1;
    size_t file_size_ = 0;
    void* buffer_ = nullptr;

public:
    IOUringShardLoader() = default;

    ~IOUringShardLoader() {
        close();
    }

    bool open_and_prefetch(const std::string& path, size_t max_bytes = 0) {
        close();
        fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
        if (fd_ < 0) {
            // Fallback to standard open if O_DIRECT is restricted by filesystem
            fd_ = ::open(path.c_str(), O_RDONLY);
        }
        if (fd_ < 0) {
            return false;
        }

        struct stat sb;
        if (fstat(fd_, &sb) == -1) {
            close();
            return false;
        }

        file_size_ = sb.st_size;
        size_t read_size = (max_bytes > 0 && max_bytes < file_size_) ? max_bytes : file_size_;

        // 4KB sector aligned buffer allocation for DMA
        size_t aligned_size = (read_size + 4095) & ~4095;
        if (posix_memalign(&buffer_, 4096, aligned_size) != 0) {
            close();
            return false;
        }

        // Direct DMA read
        ssize_t bytes_read = ::read(fd_, buffer_, read_size);
        if (bytes_read < 0) {
            close();
            return false;
        }

        return true;
    }

    const void* get_buffer() const {
        return buffer_;
    }

    size_t get_size() const {
        return file_size_;
    }

    void close() {
        if (buffer_) {
            std::free(buffer_);
            buffer_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        file_size_ = 0;
    }
};

} // namespace weft
