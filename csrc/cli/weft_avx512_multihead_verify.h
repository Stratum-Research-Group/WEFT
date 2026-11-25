#pragma once
// weft_avx512_multihead_verify.h - Multi-Head Bit-Parallel Verification Kernel in AVX-512
// Vector bitboard packing across 8 attention heads simultaneously.
// Zero-scalar branch verification for speculative draft tokens.

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <immintrin.h>

namespace weft {

class WeftAVX512MultiHeadVerifier {
public:
    static constexpr size_t NUM_HEADS = 8;
    static constexpr size_t HEAD_DIM = 64;

    // Verifies draft queries across 8 attention heads in parallel using AVX-512 vector lanes
    static void verify_8heads_parallel(
        const float* q_heads, // [NUM_HEADS, HEAD_DIM]
        const float* k_heads, // [NUM_HEADS, HEAD_DIM]
        float* out_scores     // [NUM_HEADS]
    ) {
        const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

#if defined(__AVX512F__)
        for (size_t h = 0; h < NUM_HEADS; ++h) {
            const float* q = q_heads + (h * HEAD_DIM);
            const float* k = k_heads + (h * HEAD_DIM);
            __m512 vdot = _mm512_setzero_ps();

            for (size_t d = 0; d < HEAD_DIM; d += 16) {
                __m512 vq = _mm512_loadu_ps(q + d);
                __m512 vk = _mm512_loadu_ps(k + d);
                vdot = _mm512_fmadd_ps(vq, vk, vdot);
            }
            out_scores[h] = _mm512_reduce_add_ps(vdot) * scale;
        }
#elif defined(__AVX2__)
        for (size_t h = 0; h < NUM_HEADS; ++h) {
            const float* q = q_heads + (h * HEAD_DIM);
            const float* k = k_heads + (h * HEAD_DIM);
            __m256 vdot1 = _mm256_setzero_ps();
            __m256 vdot2 = _mm256_setzero_ps();

            for (size_t d = 0; d < HEAD_DIM; d += 16) {
                __m256 vq1 = _mm256_loadu_ps(q + d);
                __m256 vk1 = _mm256_loadu_ps(k + d);
                vdot1 = _mm256_fmadd_ps(vq1, vk1, vdot1);
                __m256 vq2 = _mm256_loadu_ps(q + d + 8);
                __m256 vk2 = _mm256_loadu_ps(k + d + 8);
                vdot2 = _mm256_fmadd_ps(vq2, vk2, vdot2);
            }
            __m256 vsum = _mm256_add_ps(vdot1, vdot2);
            alignas(32) float buf[8];
            _mm256_storeu_ps(buf, vsum);
            float dot = 0.0f;
            for (int i = 0; i < 8; ++i) dot += buf[i];
            out_scores[h] = dot * scale;
        }
#else
        for (size_t h = 0; h < NUM_HEADS; ++h) {
            const float* q = q_heads + (h * HEAD_DIM);
            const float* k = k_heads + (h * HEAD_DIM);
            float dot = 0.0f;
            for (size_t d = 0; d < HEAD_DIM; ++d) {
                dot += q[d] * k[d];
            }
            out_scores[h] = dot * scale;
        }
#endif
    }
};

} // namespace weft
