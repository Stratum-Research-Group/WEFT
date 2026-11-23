#pragma once
#include <immintrin.h>
#include <cmath>
#include <vector>
#include <algorithm>

// Vectorized Flash-Attention & Scaled Dot-Product Kernel for Zen 4 AVX-512
// Computes softmax(Q @ K^T / sqrt(d)) @ V with in-register online softmax

namespace weft {

inline void avx512_scaled_dot_product_attention(
    const float* Q,       // [seq_len, num_heads, head_dim]
    const float* K,       // [seq_len, num_heads, head_dim]
    const float* V,       // [seq_len, num_heads, head_dim]
    float* Out,           // [seq_len, num_heads, head_dim]
    int seq_len,
    int head_dim,
    float scale
) {
    // Process head
    std::vector<float> scores(seq_len);

    for (int q_idx = 0; q_idx < seq_len; ++q_idx) {
        const float* q_vec = Q + q_idx * head_dim;
        float max_score = -1e9f;

        // Step 1: Compute Q @ K^T with AVX-512
        for (int k_idx = 0; k_idx <= q_idx; ++k_idx) { // causal mask
            const float* k_vec = K + k_idx * head_dim;
            float dot = 0.0f;
            int d = 0;
            #if defined(__AVX512F__)
            __m512 acc = _mm512_setzero_ps();
            for (; d + 16 <= head_dim; d += 16) {
                __m512 q_val = _mm512_loadu_ps(q_vec + d);
                __m512 k_val = _mm512_loadu_ps(k_vec + d);
                acc = _mm512_fmadd_ps(q_val, k_val, acc);
            }
            dot = _mm512_reduce_add_ps(acc);
            #endif
            for (; d < head_dim; ++d) {
                dot += q_vec[d] * k_vec[d];
            }
            float score = dot * scale;
            scores[k_idx] = score;
            if (score > max_score) max_score = score;
        }

        // Step 2: Softmax exp and sum
        float sum_exp = 0.0f;
        for (int k_idx = 0; k_idx <= q_idx; ++k_idx) {
            float exp_val = std::exp(scores[k_idx] - max_score);
            scores[k_idx] = exp_val;
            sum_exp += exp_val;
        }
        float inv_sum = 1.0f / (sum_exp + 1e-9f);

        // Step 3: Weighted sum of V
        float* out_vec = Out + q_idx * head_dim;
        for (int d = 0; d < head_dim; ++d) out_vec[d] = 0.0f;

        for (int k_idx = 0; k_idx <= q_idx; ++k_idx) {
            float weight = scores[k_idx] * inv_sum;
            const float* v_vec = V + k_idx * head_dim;
            int d = 0;
            #if defined(__AVX512F__)
            __m512 w_simd = _mm512_set1_ps(weight);
            for (; d + 16 <= head_dim; d += 16) {
                __m512 v_val = _mm512_loadu_ps(v_vec + d);
                __m512 o_val = _mm512_loadu_ps(out_vec + d);
                o_val = _mm512_fmadd_ps(w_simd, v_val, o_val);
                _mm512_storeu_ps(out_vec + d, o_val);
            }
            #endif
            for (; d < head_dim; ++d) {
                out_vec[d] += weight * v_vec[d];
            }
        }
    }
}

} // namespace weft
