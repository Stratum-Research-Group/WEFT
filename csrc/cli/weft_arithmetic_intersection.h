#pragma once
// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Arithmetic Intersection Theory on Deligne-Mumford Stacks for Cross-Attention Normalization
// Paper 207 Native Kernel Implementation

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <immintrin.h>

namespace weft {
namespace arithmetic_intersection {

struct ArithmeticCycle {
    int32_t alg_degree;
    float arch_green_norm;
};

// Compute Gillet-Soule arithmetic intersection pairing:
// (Z1 . Z2)_Ar = deg(Z1)*deg(Z2) - arch1 * arch2
inline float compute_intersection_pairing(const ArithmeticCycle& c1, const ArithmeticCycle& c2) {
    return static_cast<float>(c1.alg_degree * c2.alg_degree) - (c1.arch_green_norm * c2.arch_green_norm);
}

// Normalize cross-attention query-key logits using arithmetic intersection metrics:
// logit_ij = (Q_i . K_j)_Ar / sqrt(d) where Archimedean Green norms suppress off-diagonal noise.
inline void normalize_cross_attention_cycles(
    const std::vector<ArithmeticCycle>& queries,
    const std::vector<ArithmeticCycle>& keys,
    std::vector<float>& attention_matrix_out,
    float scale = 1.0f)
{
    const size_t M = queries.size();
    const size_t N = keys.size();
    attention_matrix_out.resize(M * N, 0.0f);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float pairing = compute_intersection_pairing(queries[i], keys[j]);
            attention_matrix_out[i * N + j] = pairing * scale;
        }
    }
}

// AVX-512 accelerated arithmetic intersection projector across uniform degree cycles
inline void batch_intersection_avx512(
    const int32_t* alg_deg_q,
    const float* arch_q,
    const int32_t* alg_deg_k,
    const float* arch_k,
    float* out_pairings,
    size_t count)
{
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= count; i += 16) {
        __m512i q_deg = _mm512_loadu_si512((const __m512i*)(alg_deg_q + i));
        __m512i k_deg = _mm512_loadu_si512((const __m512i*)(alg_deg_k + i));
        __m512 q_deg_f = _mm512_cvtepi32_ps(q_deg);
        __m512 k_deg_f = _mm512_cvtepi32_ps(k_deg);
        __m512 deg_prod = _mm512_mul_ps(q_deg_f, k_deg_f);

        __m512 q_arch = _mm512_loadu_ps(arch_q + i);
        __m512 k_arch = _mm512_loadu_ps(arch_k + i);
        __m512 arch_prod = _mm512_mul_ps(q_arch, k_arch);

        __m512 result = _mm512_sub_ps(deg_prod, arch_prod);
        _mm512_storeu_ps(out_pairings + i, result);
    }
#endif
    for (; i < count; ++i) {
        float deg_term = static_cast<float>(alg_deg_q[i] * alg_deg_k[i]);
        float arch_term = arch_q[i] * arch_k[i];
        out_pairings[i] = deg_term - arch_term;
    }
}

} // namespace arithmetic_intersection
} // namespace weft
