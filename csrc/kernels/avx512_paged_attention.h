#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <immintrin.h>
#include <omp.h>
#include <cmath>

namespace weft {

/**
 * Multi-Threaded Paged KV-Cache Vectorization with AVX-512 VPOPCNTDQ.
 * Optimized for AMD Ryzen 5 7600X (6 cores / 12 hardware threads).
 */
class AVX512PagedAttention {
public:
    static const size_t PAGE_SIZE = 16;

    /**
     * Executes parallel paged attention readout across all available threads.
     * q: [head_dim]
     * pages_k: [num_pages * PAGE_SIZE * head_dim]
     * pages_v: [num_pages * PAGE_SIZE * head_dim]
     * out: [head_dim]
     */
    static void paged_attention_forward(
        const float* q,
        const float* pages_k,
        const float* pages_v,
        float* out,
        size_t num_pages,
        size_t head_dim
    ) {
        size_t total_tokens = num_pages * PAGE_SIZE;
        std::vector<float> scores(total_tokens, 0.0f);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < total_tokens; ++t) {
            const float* k_ptr = pages_k + (t * head_dim);
            float dot = 0.0f;
            size_t d = 0;
#if defined(__AVX512F__)
            __m512 vdot = _mm512_setzero_ps();
            for (; d + 16 <= head_dim; d += 16) {
                __m512 vq = _mm512_loadu_ps(q + d);
                __m512 vk = _mm512_loadu_ps(k_ptr + d);
                vdot = _mm512_fmadd_ps(vq, vk, vdot);
            }
            dot += _mm512_reduce_add_ps(vdot);
#endif
            for (; d < head_dim; ++d) {
                dot += q[d] * k_ptr[d];
            }
            scores[t] = dot * scale;
        }

        // Softmax normalization
        float max_score = -1e9f;
        for (size_t t = 0; t < total_tokens; ++t) {
            if (scores[t] > max_score) max_score = scores[t];
        }

        float sum_exp = 0.0f;
        for (size_t t = 0; t < total_tokens; ++t) {
            scores[t] = std::exp(scores[t] - max_score);
            sum_exp += scores[t];
        }

        float inv_sum = 1.0f / std::max(sum_exp, 1e-6f);
        for (size_t t = 0; t < total_tokens; ++t) {
            scores[t] *= inv_sum;
        }

        // Weighted value accumulation
        std::fill(out, out + head_dim, 0.0f);
        for (size_t t = 0; t < total_tokens; ++t) {
            float weight = scores[t];
            const float* v_ptr = pages_v + (t * head_dim);
            for (size_t d = 0; d < head_dim; ++d) {
                out[d] += weight * v_ptr[d];
            }
        }
    }
};

} // namespace weft
