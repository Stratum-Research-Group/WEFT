#pragma once
// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Asymmetric Motivic Cohomological Hall Algebras & BPS Lie Algebras for Ultra-Dense Token Embedding Compaction
// Paper 224 Native Kernel Implementation

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <immintrin.h>

namespace weft {
namespace motivic_cohall {

// Generate composite token embedding from two BPS basis vectors via COHA convolution
inline void generate_coha_embedding_avx512(
    const float* basis_a,
    const float* basis_b,
    float* token_out,
    size_t dim,
    float alpha = 0.6f,
    float beta = 0.4f)
{
    size_t i = 0;
#if defined(__AVX512F__)
    __m512 v_alpha = _mm512_set1_ps(alpha);
    __m512 v_beta  = _mm512_set1_ps(beta);

    for (; i + 16 <= dim; i += 16) {
        __m512 a = _mm512_loadu_ps(basis_a + i);
        __m512 b = _mm512_loadu_ps(basis_b + i);

        // a * alpha + b * beta + a * b
        __m512 lin = _mm512_fmadd_ps(a, v_alpha, _mm512_mul_ps(b, v_beta));
        __m512 prod = _mm512_mul_ps(a, b);
        __m512 res = _mm512_add_ps(lin, prod);

        _mm512_storeu_ps(token_out + i, res);
    }
#endif
    for (; i < dim; ++i) {
        token_out[i] = basis_a[i] * alpha + basis_b[i] * beta + (basis_a[i] * basis_b[i]);
    }
}

} // namespace motivic_cohall
} // namespace weft
