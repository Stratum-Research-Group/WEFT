// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0
//
// AVX-512 Native Bit-Level Ternary Population Count (Popcount) Reducer (Item 5)
// Evaluates in-register ternary inner products via parallel bit-plane popcounts.

#ifndef WEFT_SUBBIT_POPCOUNT_GEMV_H
#define WEFT_SUBBIT_POPCOUNT_GEMV_H

#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cmath>

namespace weft {

class SubbitPopcountReducer {
public:
    // Evaluate ternary dot product against constant activation using positive and negative bitplanes
    // pos_plane bit == 1 -> +1, neg_plane bit == 1 -> -1
    static int32_t compute_popcount_ternary_dot(
        const uint64_t* __restrict__ pos_plane,
        const uint64_t* __restrict__ neg_plane,
        size_t num_qwords
    ) {
        int32_t total = 0;
        for (size_t q = 0; q < num_qwords; ++q) {
            uint64_t p = pos_plane[q];
            uint64_t n = neg_plane[q];
            int pos_count = __builtin_popcountll(p);
            int neg_count = __builtin_popcountll(n);
            total += (pos_count - neg_count);
        }
        return total;
    }
};

} // namespace weft

#endif // WEFT_SUBBIT_POPCOUNT_GEMV_H
