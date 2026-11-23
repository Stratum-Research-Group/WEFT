#pragma once

#include "cpu_features.h"
#include <cstdint>
#include <vector>
#include <omp.h>
#include <immintrin.h>

namespace weft {

// -------------------------------------------------------------
// 1. Scalar Reference GEMV (portable fallback)
// -------------------------------------------------------------
inline void subbit_gemv_scalar(
    const uint64_t* b_ptr,
    const uint64_t* x_ptr,
    float* out_ptr,
    const float* u_ptr,
    const float* v_x_ptr,
    int64_t rows,
    int64_t cols,
    int64_t qwords_per_row
) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        const uint64_t* row_b = b_ptr + i * qwords_per_row;
        int64_t popcnt_sum = 0;
        for (int64_t j = 0; j < qwords_per_row; ++j) {
            popcnt_sum += __builtin_popcountll(row_b[j] ^ x_ptr[j]);
        }
        float gemv_val = static_cast<float>(cols - 2 * popcnt_sum);
        out_ptr[i] = gemv_val + u_ptr[i] * (*v_x_ptr);
    }
}

// -------------------------------------------------------------
// 2. AVX2 + POPCNT GEMV
// -------------------------------------------------------------
inline void subbit_gemv_avx2(
    const uint64_t* b_ptr,
    const uint64_t* x_ptr,
    float* out_ptr,
    const float* u_ptr,
    const float* v_x_ptr,
    int64_t rows,
    int64_t cols,
    int64_t qwords_per_row
) {
    const float vx = *v_x_ptr;
    const float f_cols = static_cast<float>(cols);

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        const uint64_t* row_b = b_ptr + i * qwords_per_row;
        int64_t popcnt_sum = 0;
        int64_t j = 0;

        for (; j <= qwords_per_row - 4; j += 4) {
            uint64_t xor0 = row_b[j + 0] ^ x_ptr[j + 0];
            uint64_t xor1 = row_b[j + 1] ^ x_ptr[j + 1];
            uint64_t xor2 = row_b[j + 2] ^ x_ptr[j + 2];
            uint64_t xor3 = row_b[j + 3] ^ x_ptr[j + 3];

            popcnt_sum += _mm_popcnt_u64(xor0) +
                          _mm_popcnt_u64(xor1) +
                          _mm_popcnt_u64(xor2) +
                          _mm_popcnt_u64(xor3);
        }

        for (; j < qwords_per_row; ++j) {
            popcnt_sum += _mm_popcnt_u64(row_b[j] ^ x_ptr[j]);
        }

        float gemv_val = f_cols - 2.0f * static_cast<float>(popcnt_sum);
        out_ptr[i] = gemv_val + u_ptr[i] * vx;
    }
}

// -------------------------------------------------------------
// 3. AVX-512 + VPOPCNTDQ GEMV
// -------------------------------------------------------------
#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
inline void subbit_gemv_avx512(
    const uint64_t* b_ptr,
    const uint64_t* x_ptr,
    float* out_ptr,
    const float* u_ptr,
    const float* v_x_ptr,
    int64_t rows,
    int64_t cols,
    int64_t qwords_per_row
) {
    const float vx = *v_x_ptr;
    const float f_cols = static_cast<float>(cols);

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        const uint64_t* row_b = b_ptr + i * qwords_per_row;
        __m512i acc_popcnt = _mm512_setzero_si512();

        int64_t j = 0;
        for (; j <= qwords_per_row - 8; j += 8) {
            __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(row_b + j));
            __m512i vx_vec = _mm512_loadu_si512(reinterpret_cast<const void*>(x_ptr + j));
            __m512i vxor = _mm512_xor_si512(vb, vx_vec);
            __m512i vpop = _mm512_popcnt_epi64(vxor);
            acc_popcnt = _mm512_add_epi64(acc_popcnt, vpop);
        }

        uint64_t total_popcnt = _mm512_reduce_add_epi64(acc_popcnt);

        for (; j < qwords_per_row; ++j) {
            total_popcnt += _mm_popcnt_u64(row_b[j] ^ x_ptr[j]);
        }

        float gemv_val = f_cols - 2.0f * static_cast<float>(total_popcnt);
        out_ptr[i] = gemv_val + u_ptr[i] * vx;
    }
}
#endif

// -------------------------------------------------------------
// Unified Auto-Dispatch Entrypoint
// -------------------------------------------------------------
inline void subbit_gemv_dispatch(
    const uint64_t* b_ptr,
    const uint64_t* x_ptr,
    float* out_ptr,
    const float* u_ptr,
    const float* v_x_ptr,
    int64_t rows,
    int64_t cols
) {
    const int64_t qwords_per_row = (cols + 63) / 64;
    const CpuFeatures& feat = get_cpu_features();

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
    if (feat.optimal_isa == InstructionSet::AVX512_VPOPCNTDQ) {
        subbit_gemv_avx512(b_ptr, x_ptr, out_ptr, u_ptr, v_x_ptr, rows, cols, qwords_per_row);
        return;
    }
#endif

    if (feat.optimal_isa >= InstructionSet::AVX2 && feat.has_popcnt) {
        subbit_gemv_avx2(b_ptr, x_ptr, out_ptr, u_ptr, v_x_ptr, rows, cols, qwords_per_row);
        return;
    }

    subbit_gemv_scalar(b_ptr, x_ptr, out_ptr, u_ptr, v_x_ptr, rows, cols, qwords_per_row);
}

} // namespace weft
