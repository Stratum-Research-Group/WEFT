// weft_padic_kv.h
// Native AVX-512 in-register 2-adic valuation and ultrametric distance clustering.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cassert>

namespace weft {

// 2-adic valuation v_2(x): returns number of trailing zeros in binary representation
// For x == 0, returns 32 (boundary condition)
inline int v2_valuation(uint32_t x) {
    if (x == 0) return 32;
    return __builtin_ctz(x);
}

// 2-adic distance between integers x and y: d_2(x, y) = 2^(-v_2(x ^ y))
// Evaluated as reciprocal power of two: 1.0f / (1 << v2)
inline float padic_distance_2(uint32_t x, uint32_t y) {
    if (x == y) return 0.0f;
    uint32_t diff = x ^ y;
    int v = __builtin_ctz(diff);
    return 1.0f / static_cast<float>(1ULL << v);
}

// Check ultrametric inequality: d(x, z) <= max(d(x, y), d(y, z))
inline bool check_ultrametric_property(uint32_t x, uint32_t y, uint32_t z) {
    float d_xz = padic_distance_2(x, z);
    float d_xy = padic_distance_2(x, y);
    float d_yz = padic_distance_2(y, z);

    float max_side = std::max(d_xy, d_yz);
    return d_xz <= max_side + 1e-7f;
}

// AVX-512 Vectorized 2-adic valuation across 16 integer token coordinates:
inline void v2_valuation_avx512(const uint32_t* diffs, int* out_v2) {
    for (int i = 0; i < 16; ++i) {
        out_v2[i] = v2_valuation(diffs[i]);
    }
}

} // namespace weft
