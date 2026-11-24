#pragma once
// weft_tropical_convexity_kv.h
// Milestone Paper 232: Native AVX-512 Tropical Moduli & Tropical Convexity for KV Partitioning
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>
#include <vector>
#include <algorithm>

namespace weft {

struct TropicalTaskPartition {
    size_t task_id;
    size_t base_offset;
    size_t capacity;
    size_t total_limit;

    bool is_disjoint_from(const TropicalTaskPartition& other) const {
        return (base_offset + capacity <= other.base_offset) ||
               (other.base_offset + other.capacity <= base_offset);
    }
};

// AVX-512 kernel to evaluate tropical max-plus convex combination across cache score vectors
inline void avx512_tropical_max_plus(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    size_t count,
    float offset_a,
    float offset_b
) {
    __m512 v_off_a = _mm512_set1_ps(offset_a);
    __m512 v_off_b = _mm512_set1_ps(offset_b);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);

        __m512 va_shifted = _mm512_add_ps(va, v_off_a);
        __m512 vb_shifted = _mm512_add_ps(vb, v_off_b);

        __m512 v_max = _mm512_max_ps(va_shifted, vb_shifted);
        _mm512_storeu_ps(out + i, v_max);
    }

    for (; i < count; ++i) {
        out[i] = std::max(a[i] + offset_a, b[i] + offset_b);
    }
}

} // namespace weft
