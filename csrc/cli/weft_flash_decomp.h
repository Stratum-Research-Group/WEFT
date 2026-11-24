#pragma once
// weft_flash_decomp.h - Asymmetric Flash-Decompression & Paged KV-Cache Vector Streaming
// Milestone Paper #124
// Authors: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <array>
#include <atomic>
#include <immintrin.h>

namespace weft {

// Double-buffered vector streaming container for 2-bit compressed KV caches
class WeftFlashDecompressor {
public:
    static constexpr size_t VECTOR_WIDTH = 16; // 16 floats per AVX-512 vector
    static constexpr size_t BUFFER_TOKENS = 32; // Double-buffer partition size
    static constexpr size_t HEAD_DIM = 64;

    struct StreamingBlock {
        alignas(64) uint8_t packed_data[(HEAD_DIM * 2) / 8]; // 16 bytes per token
        float scale{1.0f};
        float zero_point{0.0f};
    };

    WeftFlashDecompressor() {
        // Pre-allocate ping-pong double buffer for zero-allocation streaming
        buffer_[0].resize(BUFFER_TOKENS * HEAD_DIM);
        buffer_[1].resize(BUFFER_TOKENS * HEAD_DIM);
    }

    // High-speed AVX-512 in-register dequantization from packed 2-bit into destination float buffer
    static void decompress_token_2bit(
        const uint8_t* packed_src,
        float scale,
        float zero_point,
        float* dest_floats
    ) {
#if defined(__AVX512F__)
        __m512 vscale = _mm512_set1_ps(scale);
        __m512 vzero  = _mm512_set1_ps(zero_point);

        // Process 16 floats per iteration (64 floats total = 4 iterations)
        for (size_t i = 0; i < 4; ++i) {
            // Read 4 bytes = 16 2-bit values
            uint32_t packed_bits;
            std::memcpy(&packed_bits, packed_src + i * 4, sizeof(uint32_t));

            alignas(64) int32_t unpacked_ints[16];
            for (size_t j = 0; j < 16; ++j) {
                unpacked_ints[j] = static_cast<int32_t>((packed_bits >> (j * 2)) & 0x03);
            }

            __m512i vi = _mm512_loadu_si512(reinterpret_cast<const void*>(unpacked_ints));
            __m512 vf = _mm512_cvtepi32_ps(vi);
            __m512 vout = _mm512_fmadd_ps(vf, vscale, vzero);

            _mm512_storeu_ps(dest_floats + i * 16, vout);
        }
#else
        for (size_t d = 0; d < HEAD_DIM; ++d) {
            size_t byte_idx = (d * 2) / 8;
            size_t shift = (d * 2) % 8;
            uint8_t q = (packed_src[byte_idx] >> shift) & 0x03;
            dest_floats[d] = static_cast<float>(q) * scale + zero_point;
        }
#endif
    }

    // Stream and decompress a partition into the target buffer index (0 or 1)
    void decompress_partition(
        const StreamingBlock* blocks,
        size_t count,
        size_t buffer_idx
    ) {
        size_t active_buf = buffer_idx % 2;
        float* dest_base = buffer_[active_buf].data();

        for (size_t t = 0; t < count && t < BUFFER_TOKENS; ++t) {
            decompress_token_2bit(
                blocks[t].packed_data,
                blocks[t].scale,
                blocks[t].zero_point,
                dest_base + t * HEAD_DIM
            );
        }
    }

    // Access active buffer slice
    const float* get_buffer(size_t buffer_idx) const {
        return buffer_[buffer_idx % 2].data();
    }

private:
    std::array<std::vector<float>, 2> buffer_;
};

} // namespace weft
