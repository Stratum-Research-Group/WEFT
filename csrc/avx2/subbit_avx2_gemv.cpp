/**
 * Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group. All rights reserved.
 * Author: Nicholas Charles Mynhardt <research@stratum.org>
 *
 * High-Performance Native C++20 AVX2 + POPCNT Sub-Bit GEMV & Matrix Operators.
 * Target: x86_64 CPUs with AVX2 + FMA + POPCNT (Intel Haswell+, Skylake, Alder/Raptor Lake; AMD Zen 1/2/3).
 * Provides bare-metal sub-bit matrix-vector multiplication with zero AVX-512 dependency.
 */

#include <torch/extension.h>
#include <immintrin.h>
#include <cstdint>
#include <vector>
#include <cmath>
#include <omp.h>

/**
 * AVX2 + POPCNT 1-Bit Bipolar Bitboard Matrix-Vector Multiplication (GEMV).
 * Computes: out = scales * (2 * popcount(x_packed XNOR w_packed) - rank)
 * x_packed: [K_words] uint64
 * b_packed: [out_features, K_words] uint64
 * scales: [out_features] float32
 * out: [out_features] float32
 */
void subbit_gemv_avx2_forward(
    const at::Tensor& x_packed,
    const at::Tensor& b_packed,
    const at::Tensor& scales,
    at::Tensor& out
) {
    TORCH_CHECK(x_packed.is_contiguous() && b_packed.is_contiguous() &&
                scales.is_contiguous() && out.is_contiguous(),
                "All input tensors must be contiguous");

    const int64_t out_features = b_packed.size(0);
    const int64_t k_words = b_packed.size(1);
    const int64_t rank = k_words * 64;

    const uint64_t* x_ptr = reinterpret_cast<const uint64_t*>(x_packed.data_ptr<int64_t>());
    const uint64_t* b_ptr = reinterpret_cast<const uint64_t*>(b_packed.data_ptr<int64_t>());
    const float* s_ptr = scales.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < out_features; ++r) {
        const uint64_t* row_b = b_ptr + r * k_words;
        int64_t total_matches = 0;

        // Process 4 words (256 bits) at a time using 64-bit hardware popcount
        int64_t w = 0;
        for (; w + 4 <= k_words; w += 4) {
            uint64_t m0 = ~(x_ptr[w + 0] ^ row_b[w + 0]);
            uint64_t m1 = ~(x_ptr[w + 1] ^ row_b[w + 1]);
            uint64_t m2 = ~(x_ptr[w + 2] ^ row_b[w + 2]);
            uint64_t m3 = ~(x_ptr[w + 3] ^ row_b[w + 3]);

            total_matches += _mm_popcnt_u64(m0) +
                             _mm_popcnt_u64(m1) +
                             _mm_popcnt_u64(m2) +
                             _mm_popcnt_u64(m3);
        }

        // Remainder loop
        for (; w < k_words; ++w) {
            uint64_t m = ~(x_ptr[w] ^ row_b[w]);
            total_matches += _mm_popcnt_u64(m);
        }

        float dot = static_cast<float>(2 * total_matches - rank);
        out_ptr[r] = dot * s_ptr[r];
    }
}


/**
 * AVX2 Batch-4 Multi-Token Register-Reused GEMM.
 * Computes: out[b, r] = scales[r] * (2 * popcount(x[b] XNOR w[r]) - rank) for b in [0, 4)
 * x_packed: [4, K_words] uint64
 * b_packed: [out_features, K_words] uint64
 * scales: [out_features] float32
 * out: [4, out_features] float32
 */
void subbit_gemm_avx2_batch4_forward(
    const at::Tensor& x_packed,
    const at::Tensor& b_packed,
    const at::Tensor& scales,
    at::Tensor& out
) {
    TORCH_CHECK(x_packed.is_contiguous() && b_packed.is_contiguous() &&
                scales.is_contiguous() && out.is_contiguous(),
                "All input tensors must be contiguous");
    TORCH_CHECK(x_packed.size(0) == 4, "x_packed batch size must be 4");

    const int64_t out_features = b_packed.size(0);
    const int64_t k_words = b_packed.size(1);
    const int64_t rank = k_words * 64;

    const uint64_t* x0 = reinterpret_cast<const uint64_t*>(x_packed.data_ptr<int64_t>());
    const uint64_t* x1 = x0 + k_words;
    const uint64_t* x2 = x1 + k_words;
    const uint64_t* x3 = x2 + k_words;

    const uint64_t* b_ptr = reinterpret_cast<const uint64_t*>(b_packed.data_ptr<int64_t>());
    const float* s_ptr = scales.data_ptr<float>();

    float* o0 = out.data_ptr<float>();
    float* o1 = o0 + out_features;
    float* o2 = o1 + out_features;
    float* o3 = o2 + out_features;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < out_features; ++r) {
        const uint64_t* row_b = b_ptr + r * k_words;
        int64_t p0 = 0, p1 = 0, p2 = 0, p3 = 0;

        for (int64_t w = 0; w < k_words; ++w) {
            uint64_t bw = row_b[w];
            p0 += _mm_popcnt_u64(~(x0[w] ^ bw));
            p1 += _mm_popcnt_u64(~(x1[w] ^ bw));
            p2 += _mm_popcnt_u64(~(x2[w] ^ bw));
            p3 += _mm_popcnt_u64(~(x3[w] ^ bw));
        }

        float scale = s_ptr[r];
        o0[r] = static_cast<float>(2 * p0 - rank) * scale;
        o1[r] = static_cast<float>(2 * p1 - rank) * scale;
        o2[r] = static_cast<float>(2 * p2 - rank) * scale;
        o3[r] = static_cast<float>(2 * p3 - rank) * scale;
    }
}


/**
 * AVX2 Vectorized Rotary Position Embedding (RoPE) Forward Pass.
 * Applies 2D Givens planar rotations in-register using 256-bit YMM floats:
 * [x0, x1] -> [x0 * cos - x1 * sin, x0 * sin + x1 * cos]
 */
void subbit_rope_avx2_forward(
    at::Tensor& x,
    const at::Tensor& cos,
    const at::Tensor& sin
) {
    TORCH_CHECK(x.is_contiguous() && cos.is_contiguous() && sin.is_contiguous(),
                "All input tensors must be contiguous");

    const int64_t total_tokens = x.numel() / x.size(-1);
    const int64_t head_dim = x.size(-1);
    const int64_t half_dim = head_dim / 2;

    float* x_ptr = x.data_ptr<float>();
    const float* cos_ptr = cos.data_ptr<float>();
    const float* sin_ptr = sin.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < total_tokens; ++t) {
        float* cur_x = x_ptr + t * head_dim;
        const float* cur_cos = cos_ptr + (t % cos.size(0)) * half_dim;
        const float* cur_sin = sin_ptr + (t % sin.size(0)) * half_dim;

        int64_t d = 0;
        // Process 8 float pairs (half_dim increments of 8)
        for (; d + 8 <= half_dim; d += 8) {
            __m256 vx0 = _mm256_loadu_ps(cur_x + d);
            __m256 vx1 = _mm256_loadu_ps(cur_x + half_dim + d);
            __m256 vcos = _mm256_loadu_ps(cur_cos + d);
            __m256 vsin = _mm256_loadu_ps(cur_sin + d);

            __m256 vout0 = _mm256_fmsub_ps(vx0, vcos, _mm256_mul_ps(vx1, vsin));
            __m256 vout1 = _mm256_fmadd_ps(vx0, vsin, _mm256_mul_ps(vx1, vcos));

            _mm256_storeu_ps(cur_x + d, vout0);
            _mm256_storeu_ps(cur_x + half_dim + d, vout1);
        }

        // Remainder loop
        for (; d < half_dim; ++d) {
            float x0_val = cur_x[d];
            float x1_val = cur_x[half_dim + d];
            float c = cur_cos[d];
            float s = cur_sin[d];

            cur_x[d] = x0_val * c - x1_val * s;
            cur_x[half_dim + d] = x0_val * s + x1_val * c;
        }
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("subbit_gemv_avx2", &subbit_gemv_avx2_forward, "AVX2 + POPCNT Sub-Bit GEMV");
    m.def("subbit_gemm_avx2_batch4", &subbit_gemm_avx2_batch4_forward, "AVX2 + POPCNT Batch-4 GEMM with Register Reuse");
    m.def("subbit_rope_avx2", &subbit_rope_avx2_forward, "AVX2 Vectorized Rotary Position Embedding (RoPE)");
}
