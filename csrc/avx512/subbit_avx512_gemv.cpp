#include <torch/extension.h>
#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cstdint>

/**
 * AVX-512 VPOPCNTDQ Accelerated Bitboard GEMV Kernel
 */
void subbit_gemv_avx512_forward(
    const at::Tensor& bitboards,      // [rows, cols / 64] uint64
    const at::Tensor& x_quant,        // [cols / 64] uint64
    at::Tensor& output,               // [rows] float32
    float scale
) {
    TORCH_CHECK(bitboards.is_contiguous(), "bitboards must be contiguous");
    TORCH_CHECK(x_quant.is_contiguous(), "x_quant must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    const int64_t rows = bitboards.size(0);
    const int64_t cols_u64 = bitboards.size(1);

    const uint64_t* b_ptr = reinterpret_cast<const uint64_t*>(bitboards.data_ptr());
    const uint64_t* x_ptr = reinterpret_cast<const uint64_t*>(x_quant.data_ptr());
    float* out_ptr = output.data_ptr<float>();

    const int64_t total_bits = cols_u64 * 64;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const uint64_t* row_b = b_ptr + r * cols_u64;
        int64_t c = 0;

        __m512i accum_popcnt = _mm512_setzero_si512();

        for (; c + 8 <= cols_u64; c += 8) {
            __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(row_b + c));
            __m512i vx = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x_ptr + c));
            __m512i vxor = _mm512_xor_si512(vb, vx);
            __m512i vcnt = _mm512_popcnt_epi64(vxor);
            accum_popcnt = _mm512_add_epi64(accum_popcnt, vcnt);
        }

        uint64_t pop_lanes[8];
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(pop_lanes), accum_popcnt);
        uint64_t total_diff_bits = 0;
        for (int i = 0; i < 8; ++i) {
            total_diff_bits += pop_lanes[i];
        }

        for (; c < cols_u64; ++c) {
            uint64_t xor_val = row_b[c] ^ x_ptr[c];
            total_diff_bits += _mm_popcnt_u64(xor_val);
        }

        int64_t net_sum = total_bits - 2 * static_cast<int64_t>(total_diff_bits);
        out_ptr[r] = static_cast<float>(net_sum) * scale;
    }
}

/**
 * AVX-512 VPOPCNTDQ Batch-4 GEMM Kernel
 */
void subbit_gemm_avx512_batch4_forward(
    const at::Tensor& bitboards,      // [rows, cols / 64] uint64
    const at::Tensor& x_quant_batch,  // [4, cols / 64] uint64
    at::Tensor& output_batch,         // [4, rows] float32
    float scale
) {
    TORCH_CHECK(bitboards.is_contiguous(), "bitboards must be contiguous");
    TORCH_CHECK(x_quant_batch.is_contiguous(), "x_quant_batch must be contiguous");
    TORCH_CHECK(output_batch.is_contiguous(), "output_batch must be contiguous");
    TORCH_CHECK(x_quant_batch.size(0) == 4, "x_quant_batch must have batch size 4");
    TORCH_CHECK(output_batch.size(0) == 4, "output_batch must have batch size 4");

    const int64_t rows = bitboards.size(0);
    const int64_t cols_u64 = bitboards.size(1);

    const uint64_t* b_ptr = reinterpret_cast<const uint64_t*>(bitboards.data_ptr());
    const uint64_t* x0_ptr = reinterpret_cast<const uint64_t*>(x_quant_batch.data_ptr());
    const uint64_t* x1_ptr = x0_ptr + cols_u64;
    const uint64_t* x2_ptr = x1_ptr + cols_u64;
    const uint64_t* x3_ptr = x2_ptr + cols_u64;

    float* out0_ptr = output_batch.data_ptr<float>();
    float* out1_ptr = out0_ptr + rows;
    float* out2_ptr = out1_ptr + rows;
    float* out3_ptr = out2_ptr + rows;

    const int64_t total_bits = cols_u64 * 64;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const uint64_t* row_b = b_ptr + r * cols_u64;
        int64_t c = 0;

        __m512i accum0 = _mm512_setzero_si512();
        __m512i accum1 = _mm512_setzero_si512();
        __m512i accum2 = _mm512_setzero_si512();
        __m512i accum3 = _mm512_setzero_si512();

        for (; c + 8 <= cols_u64; c += 8) {
            __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(row_b + c));

            __m512i vx0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x0_ptr + c));
            accum0 = _mm512_add_epi64(accum0, _mm512_popcnt_epi64(_mm512_xor_si512(vb, vx0)));

            __m512i vx1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x1_ptr + c));
            accum1 = _mm512_add_epi64(accum1, _mm512_popcnt_epi64(_mm512_xor_si512(vb, vx1)));

            __m512i vx2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x2_ptr + c));
            accum2 = _mm512_add_epi64(accum2, _mm512_popcnt_epi64(_mm512_xor_si512(vb, vx2)));

            __m512i vx3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x3_ptr + c));
            accum3 = _mm512_add_epi64(accum3, _mm512_popcnt_epi64(_mm512_xor_si512(vb, vx3)));
        }

        uint64_t diff0 = 0, diff1 = 0, diff2 = 0, diff3 = 0;
        uint64_t lanes[8];

        _mm512_storeu_si512(reinterpret_cast<__m512i*>(lanes), accum0);
        for (int i = 0; i < 8; ++i) diff0 += lanes[i];

        _mm512_storeu_si512(reinterpret_cast<__m512i*>(lanes), accum1);
        for (int i = 0; i < 8; ++i) diff1 += lanes[i];

        _mm512_storeu_si512(reinterpret_cast<__m512i*>(lanes), accum2);
        for (int i = 0; i < 8; ++i) diff2 += lanes[i];

        _mm512_storeu_si512(reinterpret_cast<__m512i*>(lanes), accum3);
        for (int i = 0; i < 8; ++i) diff3 += lanes[i];

        for (; c < cols_u64; ++c) {
            uint64_t b_val = row_b[c];
            diff0 += _mm_popcnt_u64(b_val ^ x0_ptr[c]);
            diff1 += _mm_popcnt_u64(b_val ^ x1_ptr[c]);
            diff2 += _mm_popcnt_u64(b_val ^ x2_ptr[c]);
            diff3 += _mm_popcnt_u64(b_val ^ x3_ptr[c]);
        }

        out0_ptr[r] = static_cast<float>(total_bits - 2 * static_cast<int64_t>(diff0)) * scale;
        out1_ptr[r] = static_cast<float>(total_bits - 2 * static_cast<int64_t>(diff1)) * scale;
        out2_ptr[r] = static_cast<float>(total_bits - 2 * static_cast<int64_t>(diff2)) * scale;
        out3_ptr[r] = static_cast<float>(total_bits - 2 * static_cast<int64_t>(diff3)) * scale;
    }
}

/**
 * Fused AVX-512 VPOPCNTDQ + Rank-1 SVD Correction GEMV
 */
void subbit_fused_gemv_avx512_forward(
    const at::Tensor& bitboards,      // [rows, cols / 64] uint64
    const at::Tensor& x_quant,        // [cols / 64] uint64
    const at::Tensor& u_proj,         // [rows] float32
    float v_dot_x,                   // precomputed dot product: dot(v_proj, x_fp32)
    at::Tensor& output,               // [rows] float32
    float scale
) {
    TORCH_CHECK(bitboards.is_contiguous(), "bitboards must be contiguous");
    TORCH_CHECK(x_quant.is_contiguous(), "x_quant must be contiguous");
    TORCH_CHECK(u_proj.is_contiguous(), "u_proj must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    const int64_t rows = bitboards.size(0);
    const int64_t cols_u64 = bitboards.size(1);

    const uint64_t* b_ptr = reinterpret_cast<const uint64_t*>(bitboards.data_ptr());
    const uint64_t* x_ptr = reinterpret_cast<const uint64_t*>(x_quant.data_ptr());
    const float* u_ptr = u_proj.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    const int64_t total_bits = cols_u64 * 64;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const uint64_t* row_b = b_ptr + r * cols_u64;
        int64_t c = 0;

        __m512i accum_popcnt = _mm512_setzero_si512();

        for (; c + 8 <= cols_u64; c += 8) {
            __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(row_b + c));
            __m512i vx = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x_ptr + c));
            __m512i vxor = _mm512_xor_si512(vb, vx);
            __m512i vcnt = _mm512_popcnt_epi64(vxor);
            accum_popcnt = _mm512_add_epi64(accum_popcnt, vcnt);
        }

        uint64_t pop_lanes[8];
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(pop_lanes), accum_popcnt);
        uint64_t total_diff_bits = 0;
        for (int i = 0; i < 8; ++i) {
            total_diff_bits += pop_lanes[i];
        }

        for (; c < cols_u64; ++c) {
            uint64_t xor_val = row_b[c] ^ x_ptr[c];
            total_diff_bits += _mm_popcnt_u64(xor_val);
        }

        int64_t net_sum = total_bits - 2 * static_cast<int64_t>(total_diff_bits);
        float bitboard_term = static_cast<float>(net_sum) * scale;
        float rank1_term = u_ptr[r] * v_dot_x;
        out_ptr[r] = bitboard_term + rank1_term;
    }
}

/**
 * AVX-512 Masked Ternary GEMV Kernel
 */
void subbit_ternary_gemv_avx512_forward(
    const at::Tensor& b_nonzero,      // [rows, cols / 64] uint64
    const at::Tensor& b_sign,         // [rows, cols / 64] uint64
    const at::Tensor& x_quant,        // [cols / 64] uint64
    at::Tensor& output,               // [rows] float32
    float scale
) {
    TORCH_CHECK(b_nonzero.is_contiguous(), "b_nonzero must be contiguous");
    TORCH_CHECK(b_sign.is_contiguous(), "b_sign must be contiguous");
    TORCH_CHECK(x_quant.is_contiguous(), "x_quant must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    const int64_t rows = b_nonzero.size(0);
    const int64_t cols_u64 = b_nonzero.size(1);

    const uint64_t* nz_ptr = reinterpret_cast<const uint64_t*>(b_nonzero.data_ptr());
    const uint64_t* sign_ptr = reinterpret_cast<const uint64_t*>(b_sign.data_ptr());
    const uint64_t* x_ptr = reinterpret_cast<const uint64_t*>(x_quant.data_ptr());
    float* out_ptr = output.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const uint64_t* row_nz = nz_ptr + r * cols_u64;
        const uint64_t* row_sign = sign_ptr + r * cols_u64;
        int64_t c = 0;

        __m512i accum_pos = _mm512_setzero_si512();
        __m512i accum_neg = _mm512_setzero_si512();

        for (; c + 8 <= cols_u64; c += 8) {
            __m512i vnz = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(row_nz + c));
            __m512i vsign = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(row_sign + c));
            __m512i vx = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x_ptr + c));

            __m512i vdiff = _mm512_xor_si512(vsign, vx);
            __m512i vpos_mask = _mm512_andnot_si512(vdiff, vnz);
            __m512i vneg_mask = _mm512_and_si512(vdiff, vnz);

            accum_pos = _mm512_add_epi64(accum_pos, _mm512_popcnt_epi64(vpos_mask));
            accum_neg = _mm512_add_epi64(accum_neg, _mm512_popcnt_epi64(vneg_mask));
        }

        uint64_t pos_lanes[8];
        uint64_t neg_lanes[8];
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(pos_lanes), accum_pos);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(neg_lanes), accum_neg);

        uint64_t total_pos = 0;
        uint64_t total_neg = 0;
        for (int i = 0; i < 8; ++i) {
            total_pos += pos_lanes[i];
            total_neg += neg_lanes[i];
        }

        for (; c < cols_u64; ++c) {
            uint64_t nz = row_nz[c];
            uint64_t sign = row_sign[c];
            uint64_t x = x_ptr[c];
            uint64_t diff = sign ^ x;
            total_pos += _mm_popcnt_u64((~diff) & nz);
            total_neg += _mm_popcnt_u64(diff & nz);
        }

        int64_t net_dot = static_cast<int64_t>(total_pos) - static_cast<int64_t>(total_neg);
        out_ptr[r] = static_cast<float>(net_dot) * scale;
    }
}

/**
 * AVX-512 Vectorized Rotary Position Embedding (RoPE) Kernel
 */
void subbit_rope_avx512_forward(
    const at::Tensor& x,      // [batch, heads, seq_len, head_dim] float32
    const at::Tensor& cos,    // [seq_len, head_dim / 2] float32
    const at::Tensor& sin,    // [seq_len, head_dim / 2] float32
    at::Tensor& output        // [batch, heads, seq_len, head_dim] float32
) {
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(cos.is_contiguous(), "cos must be contiguous");
    TORCH_CHECK(sin.is_contiguous(), "sin must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    const int64_t total_vectors = x.numel() / x.size(-1);
    const int64_t head_dim = x.size(-1);
    const int64_t half_dim = head_dim / 2;

    const float* x_ptr = x.data_ptr<float>();
    const float* cos_ptr = cos.data_ptr<float>();
    const float* sin_ptr = sin.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    const int64_t seq_len = cos.size(0);

    #pragma omp parallel for schedule(static)
    for (int64_t v = 0; v < total_vectors; ++v) {
        int64_t pos = v % seq_len;
        const float* c_pos = cos_ptr + pos * half_dim;
        const float* s_pos = sin_ptr + pos * half_dim;
        const float* x_vec = x_ptr + v * head_dim;
        float* out_vec = out_ptr + v * head_dim;

        int64_t i = 0;
        for (; i + 16 <= half_dim; i += 16) {
            __m512 vc = _mm512_loadu_ps(c_pos + i);
            __m512 vs = _mm512_loadu_ps(s_pos + i);

            float even_arr[16];
            float odd_arr[16];
            for (int k = 0; k < 16; ++k) {
                even_arr[k] = x_vec[2 * (i + k)];
                odd_arr[k]  = x_vec[2 * (i + k) + 1];
            }
            __m512 vx_even = _mm512_loadu_ps(even_arr);
            __m512 vx_odd  = _mm512_loadu_ps(odd_arr);

            __m512 vout_even = _mm512_sub_ps(_mm512_mul_ps(vx_even, vc), _mm512_mul_ps(vx_odd, vs));
            __m512 vout_odd  = _mm512_add_ps(_mm512_mul_ps(vx_even, vs), _mm512_mul_ps(vx_odd, vc));

            float res_even[16];
            float res_odd[16];
            _mm512_storeu_ps(res_even, vout_even);
            _mm512_storeu_ps(res_odd, vout_odd);

            for (int k = 0; k < 16; ++k) {
                out_vec[2 * (i + k)]     = res_even[k];
                out_vec[2 * (i + k) + 1] = res_odd[k];
            }
        }

        for (; i < half_dim; ++i) {
            float c = c_pos[i];
            float s = s_pos[i];
            float e = x_vec[2 * i];
            float o = x_vec[2 * i + 1];
            out_vec[2 * i]     = e * c - o * s;
            out_vec[2 * i + 1] = e * s + o * c;
        }
    }
}

/**
 * AVX-512 Numerically-Stable Softmax Kernel
 */
void subbit_softmax_avx512_forward(
    const at::Tensor& input,   // [batch_size, seq_len] float32
    at::Tensor& output         // [batch_size, seq_len] float32
) {
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    const int64_t rows = input.size(0);
    const int64_t cols = input.size(1);

    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const float* row_in = in_ptr + r * cols;
        float* row_out = out_ptr + r * cols;

        float max_val = row_in[0];
        for (int64_t c = 1; c < cols; ++c) {
            if (row_in[c] > max_val) max_val = row_in[c];
        }

        float sum_val = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float e = std::exp(row_in[c] - max_val);
            row_out[c] = e;
            sum_val += e;
        }

        float inv_sum = 1.0f / (sum_val + 1e-12f);
        int64_t c = 0;
        __m512 vinv = _mm512_set1_ps(inv_sum);
        for (; c + 16 <= cols; c += 16) {
            __m512 ve = _mm512_loadu_ps(row_out + c);
            _mm512_storeu_ps(row_out + c, _mm512_mul_ps(ve, vinv));
        }
        for (; c < cols; ++c) {
            row_out[c] *= inv_sum;
        }
    }
}


/**
 * AVX-512 In-Register Tiled Flash-Attention Forward Kernel
 * Computes O = Softmax(Q * K^T * sm_scale) * V with online max-rescaling directly in ZMM registers.
 * Q: [batch_heads, q_len, head_dim] float32
 * K: [batch_heads, kv_len, head_dim] float32
 * V: [batch_heads, kv_len, head_dim] float32
 * O: [batch_heads, q_len, head_dim] float32
 */
void subbit_flash_attention_avx512_forward(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& out,
    float sm_scale
) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && out.is_contiguous(),
                "All tensors must be contiguous");

    const int64_t batch_heads = q.size(0);
    const int64_t q_len = q.size(1);
    const int64_t head_dim = q.size(2);
    const int64_t kv_len = k.size(1);

    const float* q_ptr = q.data_ptr<float>();
    const float* k_ptr = k.data_ptr<float>();
    const float* v_ptr = v.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    const int64_t q_stride_b = q_len * head_dim;
    const int64_t kv_stride_b = kv_len * head_dim;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t bh = 0; bh < batch_heads; ++bh) {
        for (int64_t qi = 0; qi < q_len; ++qi) {
            const float* cur_q = q_ptr + bh * q_stride_b + qi * head_dim;
            const float* cur_k_base = k_ptr + bh * kv_stride_b;
            const float* cur_v_base = v_ptr + bh * kv_stride_b;
            float* cur_out = out_ptr + bh * q_stride_b + qi * head_dim;

            // Zero accumulator for this query vector
            for (int64_t d = 0; d < head_dim; ++d) {
                cur_out[d] = 0.0f;
            }

            float m_prev = -1e30f;
            float s_prev = 0.0f;

            // Online tiled loop over key/value positions
            for (int64_t ki = 0; ki < kv_len; ++ki) {
                const float* cur_k = cur_k_base + ki * head_dim;
                const float* cur_v = cur_v_base + ki * head_dim;

                // Vector dot product: q_dot_k = dot(cur_q, cur_k) * sm_scale
                __m512 vdot = _mm512_setzero_ps();
                int64_t d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    __m512 vq = _mm512_loadu_ps(cur_q + d);
                    __m512 vk = _mm512_loadu_ps(cur_k + d);
                    vdot = _mm512_fmadd_ps(vq, vk, vdot);
                }
                float score = _mm512_reduce_add_ps(vdot);
                for (; d < head_dim; ++d) {
                    score += cur_q[d] * cur_k[d];
                }
                score *= sm_scale;

                // Online softmax update:
                // m_new = max(m_prev, score)
                // alpha = exp(m_prev - m_new)
                // p_curr = exp(score - m_new)
                // s_new = s_prev * alpha + p_curr
                float m_new = (score > m_prev) ? score : m_prev;
                float alpha = std::exp(m_prev - m_new);
                float p_curr = std::exp(score - m_new);
                float s_new = s_prev * alpha + p_curr;

                // Rescale existing accumulator by alpha, then add p_curr * cur_v
                __m512 valpha = _mm512_set1_ps(alpha);
                __m512 vpcurr = _mm512_set1_ps(p_curr);
                d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    __m512 vout = _mm512_loadu_ps(cur_out + d);
                    __m512 vv = _mm512_loadu_ps(cur_v + d);
                    vout = _mm512_mul_ps(vout, valpha);
                    vout = _mm512_fmadd_ps(vpcurr, vv, vout);
                    _mm512_storeu_ps(cur_out + d, vout);
                }
                for (; d < head_dim; ++d) {
                    cur_out[d] = cur_out[d] * alpha + p_curr * cur_v[d];
                }

                m_prev = m_new;
                s_prev = s_new;
            }

            // Final normalization by 1.0 / s_final
            float inv_s = 1.0f / (s_prev + 1e-12f);
            __m512 vinv = _mm512_set1_ps(inv_s);
            int64_t d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 vout = _mm512_loadu_ps(cur_out + d);
                vout = _mm512_mul_ps(vout, vinv);
                _mm512_storeu_ps(cur_out + d, vout);
            }
            for (; d < head_dim; ++d) {
                cur_out[d] *= inv_s;
            }
        }
    }
}


/**
 * AVX-512 Fused Causal Block-Masked Flash-Attention Forward Kernel
 * Computes O = Softmax(Mask(Q * K^T * sm_scale)) * V
 * Enforces causal masking (score = -1e9 for ki > qi) directly inside vector dot products.
 * Handles head_dim = 64/128 natively with zero memory allocations.
 */
void subbit_causal_flash_attention_avx512_forward(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& out,
    float sm_scale
) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && out.is_contiguous(),
                "All tensors must be contiguous");

    const int64_t batch_heads = q.size(0);
    const int64_t q_len = q.size(1);
    const int64_t head_dim = q.size(2);
    const int64_t kv_len = k.size(1);

    const float* q_ptr = q.data_ptr<float>();
    const float* k_ptr = k.data_ptr<float>();
    const float* v_ptr = v.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    const int64_t q_stride_b = q_len * head_dim;
    const int64_t kv_stride_b = kv_len * head_dim;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t bh = 0; bh < batch_heads; ++bh) {
        for (int64_t qi = 0; qi < q_len; ++qi) {
            const float* cur_q = q_ptr + bh * q_stride_b + qi * head_dim;
            const float* cur_k_base = k_ptr + bh * kv_stride_b;
            const float* cur_v_base = v_ptr + bh * kv_stride_b;
            float* cur_out = out_ptr + bh * q_stride_b + qi * head_dim;

            for (int64_t d = 0; d < head_dim; ++d) {
                cur_out[d] = 0.0f;
            }

            float m_prev = -1e30f;
            float s_prev = 0.0f;

            // In causal attention, only keys up to the query position (or context length) are visible
            const int64_t max_k = (qi + 1 < kv_len) ? (qi + 1) : kv_len;

            for (int64_t ki = 0; ki < max_k; ++ki) {
                const float* cur_k = cur_k_base + ki * head_dim;
                const float* cur_v = cur_v_base + ki * head_dim;

                __m512 vdot = _mm512_setzero_ps();
                int64_t d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    __m512 vq = _mm512_loadu_ps(cur_q + d);
                    __m512 vk = _mm512_loadu_ps(cur_k + d);
                    vdot = _mm512_fmadd_ps(vq, vk, vdot);
                }
                float score = _mm512_reduce_add_ps(vdot);
                for (; d < head_dim; ++d) {
                    score += cur_q[d] * cur_k[d];
                }
                score *= sm_scale;

                float m_new = (score > m_prev) ? score : m_prev;
                float alpha = std::exp(m_prev - m_new);
                float p_curr = std::exp(score - m_new);
                float s_new = s_prev * alpha + p_curr;

                __m512 valpha = _mm512_set1_ps(alpha);
                __m512 vpcurr = _mm512_set1_ps(p_curr);
                d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    __m512 vout = _mm512_loadu_ps(cur_out + d);
                    __m512 vv = _mm512_loadu_ps(cur_v + d);
                    vout = _mm512_mul_ps(vout, valpha);
                    vout = _mm512_fmadd_ps(vpcurr, vv, vout);
                    _mm512_storeu_ps(cur_out + d, vout);
                }
                for (; d < head_dim; ++d) {
                    cur_out[d] = cur_out[d] * alpha + p_curr * cur_v[d];
                }

                m_prev = m_new;
                s_prev = s_new;
            }

            float inv_s = 1.0f / (s_prev + 1e-12f);
            __m512 vinv = _mm512_set1_ps(inv_s);
            int64_t d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 vout = _mm512_loadu_ps(cur_out + d);
                vout = _mm512_mul_ps(vout, vinv);
                _mm512_storeu_ps(cur_out + d, vout);
            }
            for (; d < head_dim; ++d) {
                cur_out[d] *= inv_s;
            }
        }
    }
}


/**
 * AVX-512 Masked Dynamic-Head Flash-Attention Forward Kernel
 * Supports arbitrary head dimensions (e.g. 64, 80, 96, 128) via AVX-512 tail masking.
 * Eliminates out-of-bounds reads and memory padding requirements.
 */
void subbit_dynamic_head_flash_attention_avx512_forward(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& out,
    float sm_scale
) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && out.is_contiguous(),
                "All tensors must be contiguous");

    const int64_t batch_heads = q.size(0);
    const int64_t q_len = q.size(1);
    const int64_t head_dim = q.size(2);
    const int64_t kv_len = k.size(1);

    const float* q_ptr = q.data_ptr<float>();
    const float* k_ptr = k.data_ptr<float>();
    const float* v_ptr = v.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    const int64_t q_stride_b = q_len * head_dim;
    const int64_t kv_stride_b = kv_len * head_dim;

    const int64_t tail = head_dim % 16;
    const __mmask16 tail_mask = (tail == 0) ? 0xFFFF : (static_cast<__mmask16>((1U << tail) - 1));

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int64_t bh = 0; bh < batch_heads; ++bh) {
        for (int64_t qi = 0; qi < q_len; ++qi) {
            const float* cur_q = q_ptr + bh * q_stride_b + qi * head_dim;
            const float* cur_k_base = k_ptr + bh * kv_stride_b;
            const float* cur_v_base = v_ptr + bh * kv_stride_b;
            float* cur_out = out_ptr + bh * q_stride_b + qi * head_dim;

            for (int64_t d = 0; d < head_dim; ++d) {
                cur_out[d] = 0.0f;
            }

            float m_prev = -1e30f;
            float s_prev = 0.0f;

            for (int64_t ki = 0; ki < kv_len; ++ki) {
                const float* cur_k = cur_k_base + ki * head_dim;
                const float* cur_v = cur_v_base + ki * head_dim;

                __m512 vdot = _mm512_setzero_ps();
                int64_t d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    __m512 vq = _mm512_loadu_ps(cur_q + d);
                    __m512 vk = _mm512_loadu_ps(cur_k + d);
                    vdot = _mm512_fmadd_ps(vq, vk, vdot);
                }
                if (tail > 0) {
                    __m512 vq = _mm512_maskz_loadu_ps(tail_mask, cur_q + d);
                    __m512 vk = _mm512_maskz_loadu_ps(tail_mask, cur_k + d);
                    vdot = _mm512_fmadd_ps(vq, vk, vdot);
                }
                float score = _mm512_reduce_add_ps(vdot) * sm_scale;

                float m_new = (score > m_prev) ? score : m_prev;
                float alpha = std::exp(m_prev - m_new);
                float p_curr = std::exp(score - m_new);
                float s_new = s_prev * alpha + p_curr;

                __m512 valpha = _mm512_set1_ps(alpha);
                __m512 vpcurr = _mm512_set1_ps(p_curr);
                d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    __m512 vout = _mm512_loadu_ps(cur_out + d);
                    __m512 vv = _mm512_loadu_ps(cur_v + d);
                    vout = _mm512_mul_ps(vout, valpha);
                    vout = _mm512_fmadd_ps(vpcurr, vv, vout);
                    _mm512_storeu_ps(cur_out + d, vout);
                }
                if (tail > 0) {
                    __m512 vout = _mm512_maskz_loadu_ps(tail_mask, cur_out + d);
                    __m512 vv = _mm512_maskz_loadu_ps(tail_mask, cur_v + d);
                    vout = _mm512_mul_ps(vout, valpha);
                    vout = _mm512_fmadd_ps(vpcurr, vv, vout);
                    _mm512_mask_storeu_ps(cur_out + d, tail_mask, vout);
                }

                m_prev = m_new;
                s_prev = s_new;
            }

            float inv_s = 1.0f / (s_prev + 1e-12f);
            __m512 vinv = _mm512_set1_ps(inv_s);
            int64_t d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 vout = _mm512_loadu_ps(cur_out + d);
                vout = _mm512_mul_ps(vout, vinv);
                _mm512_storeu_ps(cur_out + d, vout);
            }
            if (tail > 0) {
                __m512 vout = _mm512_maskz_loadu_ps(tail_mask, cur_out + d);
                vout = _mm512_mul_ps(vout, vinv);
                _mm512_mask_storeu_ps(cur_out + d, tail_mask, vout);
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("subbit_gemv_avx512", &subbit_gemv_avx512_forward, "AVX-512 VPOPCNTDQ Sub-Bit GEMV");
    m.def("subbit_gemm_avx512_batch4", &subbit_gemm_avx512_batch4_forward, "AVX-512 VPOPCNTDQ Batch-4 GEMM with Register Reuse");
    m.def("subbit_fused_gemv_avx512", &subbit_fused_gemv_avx512_forward, "Fused AVX-512 VPOPCNTDQ + Rank-1 SVD Correction GEMV");
    m.def("subbit_ternary_gemv_avx512", &subbit_ternary_gemv_avx512_forward, "AVX-512 Masked Ternary GEMV with Hardware Popcount");
    m.def("subbit_rope_avx512", &subbit_rope_avx512_forward, "AVX-512 Vectorized Rotary Position Embedding (RoPE)");
    m.def("subbit_softmax_avx512", &subbit_softmax_avx512_forward, "AVX-512 Numerically-Stable Vectorized Softmax");
    m.def("subbit_flash_attention_avx512", &subbit_flash_attention_avx512_forward, "AVX-512 In-Register Tiled Flash-Attention Forward");
    m.def("subbit_causal_flash_attention_avx512", &subbit_causal_flash_attention_avx512_forward, "AVX-512 Fused Causal Block-Masked Flash-Attention Forward");
    m.def("subbit_dynamic_head_flash_attention_avx512", &subbit_dynamic_head_flash_attention_avx512_forward, "AVX-512 Masked Dynamic-Head Flash-Attention Forward");
}
