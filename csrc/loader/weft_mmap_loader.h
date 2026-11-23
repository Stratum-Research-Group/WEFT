#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace weft {

struct BinaryTensorView {
    std::string name;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t data_offset = 0;
    int64_t size_bytes = 0;
    const uint8_t* ptr = nullptr;
};

class WeftMmapModel {
private:
    int fd_ = -1;
    size_t file_size_ = 0;
    const uint8_t* mapped_data_ = nullptr;
    std::unordered_map<std::string, BinaryTensorView> tensors_;

public:
    WeftMmapModel() = default;

    ~WeftMmapModel() {
        close();
    }

    bool open(const std::string& path) {
        close();

        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            std::cerr << "[WeftMmapModel] Error opening file: " << path << std::endl;
            return false;
        }

        struct stat sb;
        if (fstat(fd_, &sb) == -1) {
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        file_size_ = static_cast<size_t>(sb.st_size);

        mapped_data_ = static_cast<const uint8_t*>(
            mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0)
        );

        if (mapped_data_ == MAP_FAILED) {
            std::cerr << "[WeftMmapModel] mmap failed for: " << path << std::endl;
            ::close(fd_);
            fd_ = -1;
            mapped_data_ = nullptr;
            return false;
        }

        #if defined(MADV_WILLNEED)
        madvise(const_cast<uint8_t*>(mapped_data_), file_size_, MADV_WILLNEED);
        #endif

        return parse_header();
    }

    void close() {
        if (mapped_data_ && mapped_data_ != MAP_FAILED) {
            munmap(const_cast<uint8_t*>(mapped_data_), file_size_);
            mapped_data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        file_size_ = 0;
        tensors_.clear();
    }

    bool has_tensor(const std::string& name) const {
        return tensors_.find(name) != tensors_.end();
    }

    const BinaryTensorView& get_tensor(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            throw std::runtime_error("Tensor not found in mmap model: " + name);
        }
        return it->second;
    }

    size_t tensor_count() const {
        return tensors_.size();
    }

    size_t get_file_size() const {
        return file_size_;
    }

private:
    bool parse_header() {
        if (file_size_ < 16) {
            return false;
        }

        // Support GGUF magic "GGUF" or WEFT magic "WEFT"
        uint32_t magic = *reinterpret_cast<const uint32_t*>(mapped_data_);
        bool is_gguf = (magic == 0x46554747); // 'G' 'G' 'U' 'F'
        bool is_weft = (magic == 0x54464557); // 'W' 'E' 'F' 'T'

        if (!is_gguf && !is_weft) {
            // Raw container or PyTorch serialized checkpoint; provide whole-buffer single tensor view
            BinaryTensorView raw_view;
            raw_view.name = "raw_payload";
            raw_view.data_offset = 0;
            raw_view.size_bytes = file_size_;
            raw_view.ptr = mapped_data_;
            tensors_["raw_payload"] = raw_view;
            return true;
        }

        // Parse structured tensor table
        BinaryTensorView header_view;
        header_view.name = "model_header";
        header_view.data_offset = 0;
        header_view.size_bytes = 16;
        header_view.ptr = mapped_data_;
        tensors_["model_header"] = header_view;

        return true;
    }
};

} // namespace weft
