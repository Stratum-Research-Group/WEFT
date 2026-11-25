// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0
//
// Milestone Paper 289: Asymmetric Tropical Toric Degenerations of Flag Varieties
// for Dynamic Low-Rank KV Cache Factorization.
//
// AVX-512 SIMD kernel for low-rank key projection and reconstruction.

#ifndef WEFT_TROPICAL_FLAG_VARIETIES_H
#define WEFT_TROPICAL_FLAG_VARIETIES_H

#include <immintrin.h>
#include <cstddef>
#include <cmath>
#include <vector>

namespace weft {

class TropicalFlagLowRankEngine {
public:
    // Project full key k (dim D) into low-rank coordinates c (dim rank_r): c = U * k
    static void project_low_rank(
        const float* __restrict__ k,
        const float* __restrict__ U_matrix, // rank_r x dim
        float* __restrict__ c_out,
        size_t rank_r,
        size_t dim
    ) {
        for (size_t r = 0; r < rank_r; ++r) {
            const float* u_row = U_matrix + r * dim;
            __m512 v_dot = _mm512_setzero_ps();
            size_t i = 0;
            for (; i + 16 <= dim; i += 16) {
                __m512 vk = _mm512_loadu_ps(k + i);
                __m512 vu = _mm512_loadu_ps(u_row + i);
                v_dot = _mm512_fmadd_ps(vk, vu, v_dot);
            }
            float dot = _mm512_reduce_add_ps(v_dot);
            for (; i < dim; ++i) {
                dot += k[i] * u_row[i];
            }
            c_out[r] = dot;
        }
    }

    // Reconstruct key: k_recon = U^T * c
    static void reconstruct_key(
        const float* __restrict__ c,
        const float* __restrict__ U_matrix, // rank_r x dim
        float* __restrict__ k_recon,
        size_t rank_r,
        size_t dim
    ) {
        for (size_t i = 0; i < dim; ++i) {
            k_recon[i] = 0.0f;
        }
        for (size_t r = 0; r < rank_r; ++r) {
            float coeff = c[r];
            __m512 v_coeff = _mm512_set1_ps(coeff);
            const float* u_row = U_matrix + r * dim;
            size_t i = 0;
            for (; i + 16 <= dim; i += 16) {
                __m512 vk = _mm512_loadu_ps(k_recon + i);
                __m512 vu = _mm512_loadu_ps(u_row + i);
                vk = _mm512_fmadd_ps(v_coeff, vu, vk);
                _mm512_storeu_ps(k_recon + i, vk);
            }
            for (; i < dim; ++i) {
                k_recon[i] += coeff * u_row[i];
            }
        }
    }
};

} // namespace weft

#endif // WEFT_TROPICAL_FLAG_VARIETIES_H
