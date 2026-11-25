// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0
//
// Milestone Paper 277: Asymmetric Tropical Toric Fibers & Amoeba Volumes
// for Dynamic Layer-Skipping Speculation.
//
// AVX-512 SIMD kernel for layer residual divergence evaluation and dynamic layer skipping.

#ifndef WEFT_TROPICAL_TORIC_FIBERS_H
#define WEFT_TROPICAL_TORIC_FIBERS_H

#include <immintrin.h>
#include <cstddef>
#include <cmath>

namespace weft {

class TropicalFiberSkipEngine {
public:
    // Evaluate mean absolute divergence between consecutive layer states hl and hl1
    static float compute_layer_divergence(
        const float* __restrict__ hl,
        const float* __restrict__ hl1,
        size_t dim
    ) {
        __m512 v_acc = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 v0 = _mm512_loadu_ps(hl + i);
            __m512 v1 = _mm512_loadu_ps(hl1 + i);
            __m512 vdiff = _mm512_abs_ps(_mm512_sub_ps(v1, v0));
            v_acc = _mm512_add_ps(v_acc, vdiff);
        }
        float sum_diff = _mm512_reduce_add_ps(v_acc);
        for (; i < dim; ++i) {
            sum_diff += std::abs(hl1[i] - hl[i]);
        }
        return sum_diff / (float)dim;
    }

    // Dynamic layer skip decision: return true if layer can be skipped
    static bool should_skip_layer(
        const float* __restrict__ hl,
        const float* __restrict__ hl1,
        size_t dim,
        float threshold
    ) {
        return compute_layer_divergence(hl, hl1, dim) <= threshold;
    }
};

} // namespace weft

#endif // WEFT_TROPICAL_TORIC_FIBERS_H
