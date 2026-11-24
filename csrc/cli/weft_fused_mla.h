#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <immintrin.h>
#include "weft_mla_paged_kv.h"

namespace weft {

// Fused dot product between transformed query (512 dims) and 2-bit packed latent KV
inline float fused_mla_latent_dot(const float* transformed_query_512, const WeftMLAKVBlock& block, size_t token_offset) {
    float s = block.scale;
    float z = block.zero_point;
    float dot_accum = 0.0f;
    size_t base_byte = token_offset * 144; // 144 bytes per token for 576 dims

    size_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    __m512 v_dot = _mm512_setzero_ps();
    __m512 v_z = _mm512_set1_ps(z);

    // Process 512 latent dimensions (in 16-element chunks)
    for (; i + 16 <= 512; i += 16) {
        __m512 v_q = _mm512_loadu_ps(transformed_query_512 + i);
        float vals[16];
        for (size_t k = 0; k < 16; ++k) {
            size_t idx = i + k;
            size_t byte_idx = base_byte + (idx / 4);
            size_t shift = (idx % 4) * 2;
            uint8_t q_val = (block.packed_latent_kv[byte_idx] >> shift) & 0x03;
            vals[k] = static_cast<float>(q_val);
        }
        __m512 v_k_raw = _mm512_loadu_ps(vals);
        __m512 v_k = _mm512_sub_ps(v_k_raw, v_z);
        v_dot = _mm512_fmadd_ps(v_q, v_k, v_dot);
    }
    dot_accum += _mm512_reduce_add_ps(v_dot) * s;
#endif

    // Fallback loop
    for (; i < 512; ++i) {
        size_t byte_idx = base_byte + (i / 4);
        size_t shift = (i % 4) * 2;
        uint8_t q_val = (block.packed_latent_kv[byte_idx] >> shift) & 0x03;
        float k_val = (static_cast<float>(q_val) - z) * s;
        dot_accum += transformed_query_512[i] * k_val;
    }

    return dot_accum;
}

// Full attention logit: latent dot product + RoPE decoupled dot product
inline float fused_mla_total_score(const float* transformed_query_512, const float* query_rope_64,
                                   const WeftMLAKVBlock& block, size_t token_offset, const float* rope_key_64) {
    float latent_dot = fused_mla_latent_dot(transformed_query_512, block, token_offset);
    float rope_dot = 0.0f;
    for (size_t r = 0; r < 64; ++r) {
        rope_dot += query_rope_64[r] * rope_key_64[r];
    }
    return (latent_dot + rope_dot) * 0.0883883f; // 1 / sqrt(128)
}

} // namespace weft
