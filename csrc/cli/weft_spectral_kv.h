#pragma once
// weft_spectral_kv.h - Multi-Resolution Spectral KV-Cache Compaction
// Milestone Paper #130
// Authors: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <immintrin.h>

namespace weft {

class WeftSpectralKVEngine {
public:
    static constexpr size_t HEAD_DIM = 64;
    static constexpr float INV_SQRT2 = 0.7071067811865475f;

    WeftSpectralKVEngine() = default;

    // In-register 1D Discrete Haar Wavelet Transform over adjacent token coordinate pairs
    // Compresses 2 tokens (T_even, T_odd) into approximation (A) and detail (D):
    // A[d] = (T_even[d] + T_odd[d]) * (1 / sqrt(2))
    // D[d] = (T_even[d] - T_odd[d]) * (1 / sqrt(2))
    static void haar_forward_step(
        const float* t_even,
        const float* t_odd,
        float* out_approx,
        float* out_detail
    ) {
#if defined(__AVX512F__)
        __m512 vinv = _mm512_set1_ps(INV_SQRT2);

        for (size_t d = 0; d < HEAD_DIM; d += 16) {
            __m512 veven = _mm512_loadu_ps(t_even + d);
            __m512 vodd  = _mm512_loadu_ps(t_odd + d);

            __m512 vsum  = _mm512_add_ps(veven, vodd);
            __m512 vdiff = _mm512_sub_ps(veven, vodd);

            __m512 va = _mm512_mul_ps(vsum, vinv);
            __m512 vd = _mm512_mul_ps(vdiff, vinv);

            _mm512_storeu_ps(out_approx + d, va);
            _mm512_storeu_ps(out_detail + d, vd);
        }
#else
        for (size_t d = 0; d < HEAD_DIM; ++d) {
            out_approx[d] = (t_even[d] + t_odd[d]) * INV_SQRT2;
            out_detail[d] = (t_even[d] - t_odd[d]) * INV_SQRT2;
        }
#endif
    }

    // Inverse Haar transform:
    // T_even[d] = (A[d] + D[d]) * (1 / sqrt(2))
    // T_odd[d]  = (A[d] - D[d]) * (1 / sqrt(2))
    static void haar_inverse_step(
        const float* approx,
        const float* detail,
        float* out_even,
        float* out_odd
    ) {
#if defined(__AVX512F__)
        __m512 vinv = _mm512_set1_ps(INV_SQRT2);

        for (size_t d = 0; d < HEAD_DIM; d += 16) {
            __m512 va = _mm512_loadu_ps(approx + d);
            __m512 vd = _mm512_loadu_ps(detail + d);

            __m512 vsum  = _mm512_add_ps(va, vd);
            __m512 vdiff = _mm512_sub_ps(va, vd);

            _mm512_storeu_ps(out_even + d, _mm512_mul_ps(vsum, vinv));
            _mm512_storeu_ps(out_odd + d,  _mm512_mul_ps(vdiff, vinv));
        }
#else
        for (size_t d = 0; d < HEAD_DIM; ++d) {
            out_even[d] = (approx[d] + detail[d]) * INV_SQRT2;
            out_odd[d]  = (approx[d] - detail[d]) * INV_SQRT2;
        }
#endif
    }

    // Compute L2 signal energy: sum(x[d]^2)
    static float compute_l2_energy(const float* vec) {
        float energy = 0.0f;
        for (size_t d = 0; d < HEAD_DIM; ++d) {
            energy += vec[d] * vec[d];
        }
        return energy;
    }
};

} // namespace weft
