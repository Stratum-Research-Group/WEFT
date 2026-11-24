// weft_symplectic_grassmannian.h
// Native AVX-512 Symplectic Stiefel / Grassmannian Canonical 2-Form Inner Product & Retraction.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cassert>

namespace weft {

// Symplectic matrix frame in R^{2d x 2k}:
// Stored as [Q; P] where Q is d x 2k position block, P is d x 2k momentum block
struct SymplecticFrame {
    int d;
    int k;
    std::vector<float> Q; // Size: d * (2*k)
    std::vector<float> P; // Size: d * (2*k)

    SymplecticFrame(int dim_d, int dim_k)
        : d(dim_d), k(dim_k), Q(dim_d * (2*dim_k), 0.0f), P(dim_d * (2*dim_k), 0.0f) {}
};

// Compute symplectic inner product between column i and column j:
// omega(u, v) = u_Q^T v_P - u_P^T v_Q
inline float symplectic_inner_product_avx512(const SymplecticFrame& frame, int col_i, int col_j) {
    const int d = frame.d;
    const float* qi = &frame.Q[col_i * d];
    const float* pi = &frame.P[col_i * d];
    const float* qj = &frame.Q[col_j * d];
    const float* pj = &frame.P[col_j * d];

    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();

    int idx = 0;
    for (; idx + 16 <= d; idx += 16) {
        __m512 v_qi = _mm512_loadu_ps(&qi[idx]);
        __m512 v_pi = _mm512_loadu_ps(&pi[idx]);
        __m512 v_qj = _mm512_loadu_ps(&qj[idx]);
        __m512 v_pj = _mm512_loadu_ps(&pj[idx]);

        acc1 = _mm512_fmadd_ps(v_qi, v_pj, acc1);
        acc2 = _mm512_fmadd_ps(v_pi, v_qj, acc2);
    }

    float term1 = _mm512_reduce_add_ps(acc1);
    float term2 = _mm512_reduce_add_ps(acc2);

    for (; idx < d; ++idx) {
        term1 += qi[idx] * pj[idx];
        term2 += pi[idx] * qj[idx];
    }

    return term1 - term2;
}

} // namespace weft
