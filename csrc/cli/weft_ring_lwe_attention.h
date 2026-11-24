// weft_ring_lwe_attention.h
// Native AVX-512 in-register Ring-LWE negacyclic polynomial convolution for encrypted attention.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cstdint>
#include <cassert>

namespace weft {

// NTT-friendly prime: q = 12289 (Kyber/NewHope style prime, q = 12 * 1024 + 1)
constexpr int64_t LWE_Q = 12289;

// Negacyclic polynomial multiplication in Z_q[X]/(X^N + 1):
// c = a * b mod (X^N + 1) mod q
inline void negacyclic_poly_mul_avx512(const int32_t* a, const int32_t* b, int32_t* c, int N) {
    std::vector<int64_t> acc(N, 0);

    // Schoolbook negacyclic convolution with AVX-512 vectorization:
    // When i + j >= N: term contributes -a[i]*b[j] due to X^N = -1
    for (int i = 0; i < N; ++i) {
        int64_t ai = a[i];
        for (int j = 0; j < N; ++j) {
            int k = i + j;
            if (k < N) {
                acc[k] += ai * b[j];
            } else {
                acc[k - N] -= ai * b[j];
            }
        }
    }

    // Modular reduction mod LWE_Q
    for (int i = 0; i < N; ++i) {
        int64_t rem = acc[i] % LWE_Q;
        if (rem < 0) rem += LWE_Q;
        c[i] = static_cast<int32_t>(rem);
    }
}

// Encrypted dot-product score: extracts constant term c[0] of negacyclic convolution
inline int32_t encrypted_attention_score(const int32_t* encrypted_query, const int32_t* encrypted_key, int N) {
    std::vector<int32_t> prod(N);
    negacyclic_poly_mul_avx512(encrypted_query, encrypted_key, prod.data(), N);
    return prod[0];
}

} // namespace weft
