#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <immintrin.h>
#include "weft_paged_kv.h"

namespace weft {

// Fused bit-parallel dot product between float query and 2-bit quantized KV page
inline float fused_bit_parallel_dot(const float* query, const WeftKVBlock& block, size_t token_offset, size_t dim) {
    float s = block.scale;
    float z = block.zero_point;
    float dot_accum = 0.0f;
    size_t base_bit = token_offset * dim * 2;

    size_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    // AVX-512 accelerated path (processes 16 elements at a time)
    __m512 v_dot = _mm512_setzero_ps();
    __m512 v_z = _mm512_set1_ps(z);

    for (; i + 16 <= dim; i += 16) {
        __m512 v_q = _mm512_loadu_ps(query + i);
        // Unpack 16 2-bit values
        float vals[16];
        for (size_t k = 0; k < 16; ++k) {
            size_t bit_idx = base_bit + (i + k) * 2;
            size_t byte_idx = bit_idx / 8;
            size_t shift = bit_idx % 8;
            uint8_t q_val = (block.packed_keys[byte_idx] >> shift) & 0x03;
            vals[k] = static_cast<float>(q_val);
        }
        __m512 v_k_raw = _mm512_loadu_ps(vals);
        __m512 v_k = _mm512_sub_ps(v_k_raw, v_z);
        v_dot = _mm512_fmadd_ps(v_q, v_k, v_dot);
    }
    dot_accum += _mm512_reduce_add_ps(v_dot) * s;
#endif

    // Scalar fallback for remaining elements
    for (; i < dim; ++i) {
        size_t bit_idx = base_bit + i * 2;
        size_t byte_idx = bit_idx / 8;
        size_t shift = bit_idx % 8;
        uint8_t q_val = (block.packed_keys[byte_idx] >> shift) & 0x03;
        float k_val = (static_cast<float>(q_val) - z) * s;
        dot_accum += query[i] * k_val;
    }

    return dot_accum;
}

} // namespace weft
