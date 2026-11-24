// weft_hyperbolic_kv.h
// Native AVX-512 Lorentz Hyperboloid Minkowski Metric and Geodesic Attention Kernel.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cassert>

namespace weft {

// Lorentz Hyperboloid point in R^{d, 1}:
// Representation: x[0] is time component (x_0 > 0), x[1..d] are spatial components
// Hyperboloid condition: -x[0]^2 + sum_{i=1}^d x[i]^2 = -1 => x[0] = sqrt(1 + sum x[i]^2)
inline float compute_lorentz_time_component(const float* spatial, int d) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        __m512 v = _mm512_loadu_ps(&spatial[i]);
        acc = _mm512_fmadd_ps(v, v, acc);
    }
    float spatial_sq = _mm512_reduce_add_ps(acc);
    for (; i < d; ++i) {
        spatial_sq += spatial[i] * spatial[i];
    }
    return std::sqrt(1.0f + spatial_sq);
}

// Lorentzian Minkowski Inner Product: <u, v>_L = -u_0 * v_0 + sum_{i=1}^d u_i * v_i
inline float minkowski_inner_product_avx512(const float* u_spatial, float u0,
                                            const float* v_spatial, float v0,
                                            int d) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        __m512 vu = _mm512_loadu_ps(&u_spatial[i]);
        __m512 vv = _mm512_loadu_ps(&v_spatial[i]);
        acc = _mm512_fmadd_ps(vu, vv, acc);
    }
    float spatial_dot = _mm512_reduce_add_ps(acc);
    for (; i < d; ++i) {
        spatial_dot += u_spatial[i] * v_spatial[i];
    }
    return -u0 * v0 + spatial_dot;
}

// Hyperbolic Geodesic Distance: d_H(u, v) = arcosh(-<u, v>_L)
inline float hyperbolic_geodesic_distance(const float* u_spatial, float u0,
                                          const float* v_spatial, float v0,
                                          int d) {
    float inner_prod = minkowski_inner_product_avx512(u_spatial, u0, v_spatial, v0, d);
    float arg = -inner_prod;
    if (arg < 1.0f) arg = 1.0f; // Clamp machine epsilon
    return std::acosh(arg);
}

} // namespace weft
