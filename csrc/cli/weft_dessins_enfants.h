#pragma once
// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Asymmetric Absolute Galois Groups & Grothendieck's Dessins d'Enfants for Bipartite Attention Permutations
// Paper 210 Native Kernel Implementation

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <immintrin.h>

namespace weft {
namespace dessins_enfants {

struct DessinPermutation {
    std::vector<uint32_t> sigma_0;   // Rotation around black vertices
    std::vector<uint32_t> sigma_1;   // Rotation around white vertices
    std::vector<uint32_t> sigma_inf; // Rotation around faces: (sigma_0 * sigma_1)^(-1)
};

// Permute attention token embeddings using dessin permutation indices
inline void permute_dessin_channels(
    const float* src_channels,
    const uint32_t* perm_indices,
    float* dst_channels,
    size_t count)
{
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= count; i += 16) {
        __m512i idx = _mm512_loadu_si512((const __m512i*)(perm_indices + i));
        __m512 val = _mm512_i32gather_ps(idx, src_channels, 4);
        _mm512_storeu_ps(dst_channels + i, val);
    }
#endif
    for (; i < count; ++i) {
        dst_channels[i] = src_channels[perm_indices[i]];
    }
}

// Compute the face monodromy permutation: sigma_inf = (sigma_0 * sigma_1)^(-1)
inline std::vector<uint32_t> compute_sigma_infinity(
    const std::vector<uint32_t>& sigma_0,
    const std::vector<uint32_t>& sigma_1)
{
    size_t n = sigma_0.size();
    std::vector<uint32_t> prod(n);
    std::vector<uint32_t> inv(n);

    // prod = sigma_0 * sigma_1: prod[i] = sigma_0[sigma_1[i]]
    for (size_t i = 0; i < n; ++i) {
        prod[i] = sigma_0[sigma_1[i]];
    }
    // inv = prod^(-1)
    for (size_t i = 0; i < n; ++i) {
        inv[prod[i]] = static_cast<uint32_t>(i);
    }
    return inv;
}

} // namespace dessins_enfants
} // namespace weft
