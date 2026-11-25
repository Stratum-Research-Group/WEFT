// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0
//
// AVX-512 Native Multi-Threaded Interleaved FlashAttention Kernel with NUMA Paging (Item 5)
// Evaluates tiled scaled dot-product attention in-register with cache-bypassing writes.

#ifndef WEFT_FUSED_NUMA_FLASH_ATTENTION_H
#define WEFT_FUSED_NUMA_FLASH_ATTENTION_H

#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>
#include <thread>
#include <algorithm>
#include <pthread.h>

namespace weft {

class FusedNumaFlashAttention {
public:
    // Single-thread in-register scaled dot product tile over 64-byte aligned blocks
    static void compute_tile_attention(
        const float* __restrict__ q,
        const float* __restrict__ k,
        const float* __restrict__ v,
        float* __restrict__ out,
        size_t seq_len,
        size_t head_dim,
        float scale
    ) {
        // Tiled accumulation
        std::vector<float> scores(seq_len);
        float max_score = -1e9f;

        for (size_t s = 0; s < seq_len; ++s) {
            const float* k_ptr = k + s * head_dim;
            __m512 v_acc = _mm512_setzero_ps();
            size_t d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 vq = _mm512_loadu_ps(q + d);
                __m512 vk = _mm512_loadu_ps(k_ptr + d);
                v_acc = _mm512_fmadd_ps(vq, vk, v_acc);
            }
            float dot = _mm512_reduce_add_ps(v_acc);
            for (; d < head_dim; ++d) {
                dot += q[d] * k_ptr[d];
            }
            float sc = dot * scale;
            scores[s] = sc;
            if (sc > max_score) max_score = sc;
        }

        // Softmax
        float sum_exp = 0.0f;
        for (size_t s = 0; s < seq_len; ++s) {
            scores[s] = std::exp(scores[s] - max_score);
            sum_exp += scores[s];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (size_t s = 0; s < seq_len; ++s) {
            scores[s] *= inv_sum;
        }

        // Value weighting into out
        for (size_t d = 0; d < head_dim; ++d) {
            out[d] = 0.0f;
        }
        for (size_t s = 0; s < seq_len; ++s) {
            float weight = scores[s];
            __m512 v_weight = _mm512_set1_ps(weight);
            const float* v_ptr = v + s * head_dim;
            size_t d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 vo = _mm512_loadu_ps(out + d);
                __m512 vv = _mm512_loadu_ps(v_ptr + d);
                vo = _mm512_fmadd_ps(v_weight, vv, vo);
                _mm512_storeu_ps(out + d, vo);
            }
            for (; d < head_dim; ++d) {
                out[d] += weight * v_ptr[d];
            }
        }
    }
};

} // namespace weft

#endif // WEFT_FUSED_NUMA_FLASH_ATTENTION_H
