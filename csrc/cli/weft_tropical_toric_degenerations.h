// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0
//
// Milestone Paper 274: Asymmetric Tropical Toric Degenerations & Polyhedral Complexes
// for Real-Time KV Cache Compaction.
//
// AVX-512 SIMD kernel for evaluating tropical polyhedral distance and merging KV pages.

#ifndef WEFT_TROPICAL_TORIC_DEGENERATIONS_H
#define WEFT_TROPICAL_TORIC_DEGENERATIONS_H

#include <immintrin.h>
#include <cstddef>
#include <cmath>
#include <algorithm>

namespace weft {

class TropicalToricCompactor {
public:
    // Evaluate Chebyshev / L_infty tropical distance between two key vectors: max_i |k_a[i] - k_b[i]|
    static float compute_tropical_distance(
        const float* __restrict__ k_a,
        const float* __restrict__ k_b,
        size_t dim
    ) {
        __m512 v_max = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 va = _mm512_loadu_ps(k_a + i);
            __m512 vb = _mm512_loadu_ps(k_b + i);
            __m512 vdiff = _mm512_abs_ps(_mm512_sub_ps(va, vb));
            v_max = _mm512_max_ps(v_max, vdiff);
        }
        float max_val = _mm512_reduce_max_ps(v_max);
        for (; i < dim; ++i) {
            float diff = std::abs(k_a[i] - k_b[i]);
            if (diff > max_val) {
                max_val = diff;
            }
        }
        return max_val;
    }

    // Merge two key vectors into a polyhedral representative: k_out = 0.5 * (k_a + k_b)
    static void merge_tropical_keys(
        const float* __restrict__ k_a,
        const float* __restrict__ k_b,
        float* __restrict__ k_out,
        size_t dim
    ) {
        __m512 v_half = _mm512_set1_ps(0.5f);
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 va = _mm512_loadu_ps(k_a + i);
            __m512 vb = _mm512_loadu_ps(k_b + i);
            __m512 vsum = _mm512_add_ps(va, vb);
            __m512 vres = _mm512_mul_ps(vsum, v_half);
            _mm512_storeu_ps(k_out + i, vres);
        }
        for (; i < dim; ++i) {
            k_out[i] = 0.5f * (k_a[i] + k_b[i]);
        }
    }
};

} // namespace weft

#endif // WEFT_TROPICAL_TORIC_DEGENERATIONS_H
