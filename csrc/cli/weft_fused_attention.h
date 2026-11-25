#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <immintrin.h>
#include <omp.h>
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

// Native AVX-512 Flash-Decoding over Paged KV Blocks
// Evaluates multi-head attention across all paged KV blocks with online softmax
inline void paged_flash_decoding_avx512(
    const float* query,                     // [num_heads, head_dim]
    const std::vector<WeftKVBlock>& blocks, // Paged KV chain
    float* out,                             // [num_heads, head_dim] output
    size_t num_heads,
    size_t head_dim,
    float scale = 0.125f                    // 1.0 / sqrt(head_dim)
) {
    if (blocks.empty()) return;

    size_t total_tokens = 0;
    for (const auto& b : blocks) total_tokens += b.num_tokens;
    if (total_tokens == 0) return;

    #pragma omp parallel for schedule(static)
    for (size_t h = 0; h < num_heads; ++h) {
        const float* q_h = query + h * head_dim;
        float* out_h = out + h * head_dim;

        // Online softmax state
        float m_prev = -1e9f;
        float l_prev = 0.0f;
        std::vector<float> acc(head_dim, 0.0f);

        for (const auto& block : blocks) {
            float s = block.scale;
            float z = block.zero_point;

            for (size_t t = 0; t < block.num_tokens; ++t) {
                // Compute Q * K_t^T
                float score = fused_bit_parallel_dot(q_h, block, t, head_dim) * scale;

                // Online softmax update
                float m_new = std::max(m_prev, score);
                float exp_prev = std::exp(m_prev - m_new);
                float exp_curr = std::exp(score - m_new);
                float l_new = l_prev * exp_prev + exp_curr;

                // Rescale accumulator and add V_t
                size_t base_v_bit = t * head_dim * 2;
                for (size_t d = 0; d < head_dim; ++d) {
                    size_t bit_idx = base_v_bit + d * 2;
                    size_t byte_idx = bit_idx / 8;
                    size_t shift = bit_idx % 8;
                    uint8_t q_val = (block.packed_values[byte_idx] >> shift) & 0x03;
                    float v_val = (static_cast<float>(q_val) - z) * s;

                    acc[d] = acc[d] * exp_prev + exp_curr * v_val;
                }

                m_prev = m_new;
                l_prev = l_new;
            }
        }

        // Final normalization
        float inv_l = (l_prev > 1e-7f) ? (1.0f / l_prev) : 1.0f;
        for (size_t d = 0; d < head_dim; ++d) {
            out_h[d] = acc[d] * inv_l;
        }
    }
}

} // namespace weft
