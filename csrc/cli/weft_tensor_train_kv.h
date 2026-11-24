// weft_tensor_train_kv.h
// Native AVX-512 in-register Tensor Train core contractions for compressed multi-head KV cache.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cassert>

namespace weft {

struct TTCore {
    int r_in;
    int mode_dim;
    int r_out;
    std::vector<float> data; // Size: r_in * mode_dim * r_out

    TTCore(int rin, int mode, int rout)
        : r_in(rin), mode_dim(mode), r_out(rout), data(rin * mode * rout, 0.0f) {}

    float& at(int i, int j, int k) {
        return data[(i * mode_dim + j) * r_out + k];
    }

    const float& at(int i, int j, int k) const {
        return data[(i * mode_dim + j) * r_out + k];
    }
};

// Contract input vector x (dimension mode_dim) with TT core G:
// out_vec[r_out] = sum_j x[j] * G(0, j, k) (for first core where r_in = 1)
inline void tt_contract_core1_avx512(const float* x, const TTCore& core, float* out) {
    assert(core.r_in == 1);
    const int mode = core.mode_dim;
    const int rout = core.r_out;

    std::memset(out, 0, rout * sizeof(float));

    for (int k = 0; k < rout; ++k) {
        __m512 acc = _mm512_setzero_ps();
        int j = 0;
        for (; j + 16 <= mode; j += 16) {
            __m512 vx = _mm512_loadu_ps(&x[j]);
            // Load core elements G(0, j..j+15, k)
            alignas(64) float g_slice[16];
            for (int s = 0; s < 16; ++s) {
                g_slice[s] = core.at(0, j + s, k);
            }
            __m512 vg = _mm512_load_ps(g_slice);
            acc = _mm512_fmadd_ps(vx, vg, acc);
        }
        float sum = _mm512_reduce_add_ps(acc);
        for (; j < mode; ++j) {
            sum += x[j] * core.at(0, j, k);
        }
        out[k] = sum;
    }
}

// Full 2-core contraction for a multi-head key tensor slice:
// x (dim n1), core1 (1, n1, r), core2 (r, n2, 1) -> reconstructed dot product against query q (dim n2)
inline float tt_contract_attention_score(const float* x, const TTCore& c1, const TTCore& c2, const float* q) {
    assert(c1.r_in == 1);
    assert(c1.r_out == c2.r_in);
    assert(c2.r_out == 1);

    const int r = c1.r_out;
    std::vector<float> intermediate(r);
    tt_contract_core1_avx512(x, c1, intermediate.data());

    // Second contraction with c2 and query vector q:
    // score = sum_k intermediate[k] * (sum_j q[j] * c2(k, j, 0))
    float total_score = 0.0f;
    for (int k = 0; k < r; ++k) {
        __m512 acc = _mm512_setzero_ps();
        int j = 0;
        const int mode2 = c2.mode_dim;
        for (; j + 16 <= mode2; j += 16) {
            __m512 vq = _mm512_loadu_ps(&q[j]);
            alignas(64) float g2_slice[16];
            for (int s = 0; s < 16; ++s) {
                g2_slice[s] = c2.at(k, j + s, 0);
            }
            __m512 vg2 = _mm512_load_ps(g2_slice);
            acc = _mm512_fmadd_ps(vq, vg2, acc);
        }
        float sub_score = _mm512_reduce_add_ps(acc);
        for (; j < mode2; ++j) {
            sub_score += q[j] * c2.at(k, j, 0);
        }
        total_score += intermediate[k] * sub_score;
    }

    return total_score;
}

} // namespace weft
