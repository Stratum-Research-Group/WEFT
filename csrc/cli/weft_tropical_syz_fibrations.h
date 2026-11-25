// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0
//
// Milestone Paper 283: Asymmetric Tropical Toric Fibrations & SYZ Fibrations
// for Distributed Edge Attention Sharding.
//
// AVX-512 SIMD kernel for shard accumulation and weighted reduction.

#ifndef WEFT_TROPICAL_SYZ_FIBRATIONS_H
#define WEFT_TROPICAL_SYZ_FIBRATIONS_H

#include <immintrin.h>
#include <cstddef>
#include <cmath>

namespace weft {

class SYZShardAccumulator {
public:
    // Accumulate weighted shard vector: y_out[i] += weight * shard[i]
    static void accumulate_shard(
        const float* __restrict__ shard,
        float weight,
        float* __restrict__ y_out,
        size_t dim
    ) {
        __m512 v_weight = _mm512_set1_ps(weight);
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m512 vs = _mm512_loadu_ps(shard + i);
            __m512 vy = _mm512_loadu_ps(y_out + i);
            vy = _mm512_fmadd_ps(vs, v_weight, vy);
            _mm512_storeu_ps(y_out + i, vy);
        }
        for (; i < dim; ++i) {
            y_out[i] += weight * shard[i];
        }
    }
};

} // namespace weft

#endif // WEFT_TROPICAL_SYZ_FIBRATIONS_H
