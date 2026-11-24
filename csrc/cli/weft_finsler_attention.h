// weft_finsler_attention.h
// Native AVX-512 Randers Finsler Metric for Non-Reversible Causal Attention.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cassert>

namespace weft {

// Compute non-reversible Randers Finsler distance:
// F(u, v) = sqrt(sum (v_i - u_i)^2) + sum b_i * (v_i - u_i)
// Note: F(u, v) != F(v, u) when drift vector b != 0
inline float randers_distance_avx512(const float* u, const float* v, const float* b, int dim) {
    __m512 acc_sq = _mm512_setzero_ps();
    __m512 acc_drift = _mm512_setzero_ps();

    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 vu = _mm512_loadu_ps(&u[i]);
        __m512 vv = _mm512_loadu_ps(&v[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);

        __m512 diff = _mm512_sub_ps(vv, vu); // (v - u)
        acc_sq = _mm512_fmadd_ps(diff, diff, acc_sq);
        acc_drift = _mm512_fmadd_ps(vb, diff, acc_drift);
    }

    float norm_sq = _mm512_reduce_add_ps(acc_sq);
    float drift = _mm512_reduce_add_ps(acc_drift);

    for (; i < dim; ++i) {
        float diff = v[i] - u[i];
        norm_sq += diff * diff;
        drift += b[i] * diff;
    }

    float alpha = std::sqrt(norm_sq);
    return alpha + drift;
}

// Causal asymmetric attention score: exp(-gamma * F(k, q))
inline float finsler_causal_attention_score(const float* k, const float* q, const float* b, int dim, float gamma = 1.0f) {
    float dist = randers_distance_avx512(k, q, b, dim);
    if (dist < 0.0f) dist = 0.0f; // Positive-definiteness guarantee
    return std::exp(-gamma * dist);
}

} // namespace weft
