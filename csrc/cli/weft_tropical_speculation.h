// weft_tropical_speculation.h
// Native AVX-512 in-register Tropical Min-Plus Semiring Matrix-Vector Product.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <cassert>

namespace weft {

// Tropical Matrix-Vector Product: (A (x) x)_i = min_j (A_{ij} + x_j)
inline void tropical_gemv_avx512(const float* A, const float* x, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* a_row = &A[r * cols];
        __m512 v_min = _mm512_set1_ps(std::numeric_limits<float>::infinity());

        int c = 0;
        for (; c + 16 <= cols; c += 16) {
            __m512 va = _mm512_loadu_ps(&a_row[c]);
            __m512 vx = _mm512_loadu_ps(&x[c]);
            __m512 vsum = _mm512_add_ps(va, vx);
            v_min = _mm512_min_ps(v_min, vsum);
        }

        float min_val = _mm512_reduce_min_ps(v_min);
        for (; c < cols; ++c) {
            min_val = std::min(min_val, a_row[c] + x[c]);
        }
        out[r] = min_val;
    }
}

// Find optimal token candidate index: argmin_i out[i]
inline int tropical_best_candidate(const float* out, int size) {
    int best_idx = 0;
    float best_cost = out[0];
    for (int i = 1; i < size; ++i) {
        if (out[i] < best_cost) {
            best_cost = out[i];
            best_idx = i;
        }
    }
    return best_idx;
}

} // namespace weft
