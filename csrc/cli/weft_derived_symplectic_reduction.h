// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0
//
// Milestone Paper 276: Asymmetric Derived Symplectic Reductions & Moment Maps
// for Gradient-Free Frozen Weight Orthogonalization.
//
// AVX-512 SIMD kernel for symplectic projection and moment map reduction.

#ifndef WEFT_DERIVED_SYMPLECTIC_REDUCTION_H
#define WEFT_DERIVED_SYMPLECTIC_REDUCTION_H

#include <immintrin.h>
#include <cstddef>
#include <cmath>

namespace weft {

class SymplecticReductionEngine {
public:
    // Project vector v onto the orthogonal complement of vector w: v_out = v - (v.w / ||w||^2) * w
    static void project_orthogonal(
        const float* __restrict__ v,
        const float* __restrict__ w,
        float* __restrict__ v_out,
        size_t dim,
        float epsilon = 1e-8f
    ) {
        __m512 v_dot = _mm512_setzero_ps();
        __m512 v_norm_w = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 vv = _mm512_loadu_ps(v + i);
            __m512 vw = _mm512_loadu_ps(w + i);
            v_dot = _mm512_fmadd_ps(vv, vw, v_dot);
            v_norm_w = _mm512_fmadd_ps(vw, vw, v_norm_w);
        }
        float dot = _mm512_reduce_add_ps(v_dot);
        float norm_w = _mm512_reduce_add_ps(v_norm_w);
        for (; i < dim; ++i) {
            dot += v[i] * w[i];
            norm_w += w[i] * w[i];
        }

        float coeff = dot / (norm_w + epsilon);
        __m512 v_coeff = _mm512_set1_ps(coeff);

        i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 vv = _mm512_loadu_ps(v + i);
            __m512 vw = _mm512_loadu_ps(w + i);
            __m512 v_proj = _mm512_fnmadd_ps(v_coeff, vw, vv);
            _mm512_storeu_ps(v_out + i, v_proj);
        }
        for (; i < dim; ++i) {
            v_out[i] = v[i] - coeff * w[i];
        }
    }

    // Compute residual inner product to verify moment map constraint <v_proj, w> == 0
    static float compute_residual_dot(
        const float* __restrict__ v_out,
        const float* __restrict__ w,
        size_t dim
    ) {
        __m512 v_dot = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 vv = _mm512_loadu_ps(v_out + i);
            __m512 vw = _mm512_loadu_ps(w + i);
            v_dot = _mm512_fmadd_ps(vv, vw, v_dot);
        }
        float dot = _mm512_reduce_add_ps(v_dot);
        for (; i < dim; ++i) {
            dot += v_out[i] * w[i];
        }
        return std::abs(dot);
    }
};

} // namespace weft

#endif // WEFT_DERIVED_SYMPLECTIC_REDUCTION_H
