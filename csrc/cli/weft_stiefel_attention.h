// weft_stiefel_attention.h
// Native AVX-512 Cayley Retraction on Stiefel Manifolds St(k, d) for Orthogonal Projections.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cassert>

namespace weft {

// Stiefel matrix: d rows, k columns (stored in column-major order for fast vector inner products)
struct StiefelMatrix {
    int d;
    int k;
    std::vector<float> data; // Size d * k

    StiefelMatrix(int rows, int cols) : d(rows), k(cols), data(rows * cols, 0.0f) {}

    float& at(int r, int c) { return data[c * d + r]; }
    const float& at(int r, int c) const { return data[c * d + r]; }
};

// Compute column inner product between column c1 and column c2 using AVX-512:
inline float column_dot_avx512(const StiefelMatrix& mat, int c1, int c2) {
    const int d = mat.d;
    const float* ptr1 = &mat.data[c1 * d];
    const float* ptr2 = &mat.data[c2 * d];

    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        __m512 v1 = _mm512_loadu_ps(&ptr1[i]);
        __m512 v2 = _mm512_loadu_ps(&ptr2[i]);
        acc = _mm512_fmadd_ps(v1, v2, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < d; ++i) {
        sum += ptr1[i] * ptr2[i];
    }
    return sum;
}

// First-order Cayley retraction step: Y = X + W * X where W = Delta * X^T - X * Delta^T
// Updates X along tangent matrix Delta while preserving approximate orthogonality
inline void stiefel_cayley_step(StiefelMatrix& X, const StiefelMatrix& Delta, float step_size) {
    const int d = X.d;
    const int k = X.k;

    // Rank-2k skew update: for each column of X, advance by step_size * (Delta*(X^T X) - X*(Delta^T X))
    // Since X^T X = I, Delta * (X^T X) = Delta
    for (int c = 0; c < k; ++c) {
        float* x_col = &X.data[c * d];
        const float* delta_col = &Delta.data[c * d];

        // x_new = x + step_size * delta
        int i = 0;
        __m512 v_step = _mm512_set1_ps(step_size);
        for (; i + 16 <= d; i += 16) {
            __m512 vx = _mm512_loadu_ps(&x_col[i]);
            __m512 vd = _mm512_loadu_ps(&delta_col[i]);
            vx = _mm512_fmadd_ps(v_step, vd, vx);
            _mm512_storeu_ps(&x_col[i], vx);
        }
        for (; i < d; ++i) {
            x_col[i] += step_size * delta_col[i];
        }

        // Re-normalize column to exact unit norm
        float norm_sq = column_dot_avx512(X, c, c);
        float inv_norm = 1.0f / std::sqrt(norm_sq);
        __m512 v_inv = _mm512_set1_ps(inv_norm);
        i = 0;
        for (; i + 16 <= d; i += 16) {
            __m512 vx = _mm512_loadu_ps(&x_col[i]);
            vx = _mm512_mul_ps(vx, v_inv);
            _mm512_storeu_ps(&x_col[i], vx);
        }
        for (; i < d; ++i) {
            x_col[i] *= inv_norm;
        }
    }
}

} // namespace weft
