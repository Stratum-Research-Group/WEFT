#pragma once
// weft_cache_oblivious_gemm.h - Polyhedral Cache-Oblivious Micro-Tiling GEMM
// Milestone Paper #133
// Authors: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

class WeftCacheObliviousGEMM {
public:
    static constexpr size_t BASE_M = 8;
    static constexpr size_t BASE_N = 16; // AVX-512 vector width (16 floats)

    WeftCacheObliviousGEMM() = default;

    // Base-case micro-kernel: C[m, n] += A[m, k] * dequant(W_packed[k, n])
    // Accumulates in 16-wide AVX-512 vector registers with zero heap overhead
    static void micro_kernel(
        const float* A,
        const uint8_t* W_packed,
        float* C,
        size_t M, size_t N, size_t K,
        size_t lda, size_t ldw, size_t ldc,
        float scale, float zero_point
    ) {
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; n += 16) {
#if defined(__AVX512F__)
                __m512 vc = _mm512_loadu_ps(C + m * ldc + n);
                __m512 vscale = _mm512_set1_ps(scale);
                __m512 vzero  = _mm512_set1_ps(zero_point);

                for (size_t k = 0; k < K; ++k) {
                    float a_val = A[m * lda + k];
                    __m512 va = _mm512_set1_ps(a_val);

                    // Dequantize 16 weights on the fly (4 bytes packed 2-bit)
                    size_t byte_offset = k * ldw + (n / 4);
                    uint32_t packed_bits = *reinterpret_cast<const uint32_t*>(W_packed + byte_offset);

                    alignas(64) float unpacked_w[16];
                    for (size_t j = 0; j < 16; ++j) {
                        uint8_t q = (packed_bits >> (j * 2)) & 0x03;
                        unpacked_w[j] = static_cast<float>(q);
                    }

                    __m512 vw_raw = _mm512_loadu_ps(unpacked_w);
                    __m512 vw = _mm512_fmadd_ps(vw_raw, vscale, vzero);

                    vc = _mm512_fmadd_ps(va, vw, vc);
                }

                _mm512_storeu_ps(C + m * ldc + n, vc);
#else
                for (size_t j = 0; j < 16 && (n + j) < N; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        size_t byte_offset = k * ldw + ((n + j) / 4);
                        size_t shift = ((n + j) % 4) * 2;
                        uint8_t q = (W_packed[byte_offset] >> shift) & 0x03;
                        float w = static_cast<float>(q) * scale + zero_point;
                        sum += A[m * lda + k] * w;
                    }
                    C[m * ldc + n + j] += sum;
                }
#endif
            }
        }
    }

    // Recursive cache-oblivious matrix multiplication
    static void gemm_recursive(
        const float* A,
        const uint8_t* W_packed,
        float* C,
        size_t M, size_t N, size_t K,
        size_t lda, size_t ldw, size_t ldc,
        float scale, float zero_point
    ) {
        // Base case threshold
        if (M <= BASE_M && N <= BASE_N) {
            micro_kernel(A, W_packed, C, M, N, K, lda, ldw, ldc, scale, zero_point);
            return;
        }

        // Halve the largest dimension
        if (M >= N && M >= K && M > BASE_M) {
            size_t M1 = M / 2;
            gemm_recursive(A, W_packed, C, M1, N, K, lda, ldw, ldc, scale, zero_point);
            gemm_recursive(A + M1 * lda, W_packed, C + M1 * ldc, M - M1, N, K, lda, ldw, ldc, scale, zero_point);
        } else if (N >= K && N > BASE_N) {
            size_t N1 = (N / 2 / 16) * 16;
            if (N1 == 0) N1 = 16;
            gemm_recursive(A, W_packed, C, M, N1, K, lda, ldw, ldc, scale, zero_point);
            gemm_recursive(A, W_packed + (N1 / 4), C + N1, M, N - N1, K, lda, ldw, ldc, scale, zero_point);
        } else {
            size_t K1 = K / 2;
            if (K1 == 0) {
                micro_kernel(A, W_packed, C, M, N, K, lda, ldw, ldc, scale, zero_point);
                return;
            }
            gemm_recursive(A, W_packed, C, M, N, K1, lda, ldw, ldc, scale, zero_point);
            gemm_recursive(A + K1, W_packed + K1 * ldw, C, M, N, K - K1, lda, ldw, ldc, scale, zero_point);
        }
    }
};

} // namespace weft
