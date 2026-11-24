#pragma once
// weft_fused_rope.h - Asymmetric Rotary Position Embedding In-Register Compaction
// Milestone Paper #129
// Authors: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <immintrin.h>

namespace weft {

class WeftFusedRoPE {
public:
    static constexpr size_t HEAD_DIM = 64;
    static constexpr size_t HALF_DIM = HEAD_DIM / 2;

    WeftFusedRoPE(float base_freq = 10000.0f) {
        inv_freq_.resize(HALF_DIM);
        for (size_t i = 0; i < HALF_DIM; ++i) {
            inv_freq_[i] = 1.0f / std::pow(base_freq, static_cast<float>(2 * i) / static_cast<float>(HEAD_DIM));
        }
    }

    // Apply RoPE in-register to 2-bit unpacked vector coordinates:
    // x0' = x0 * cos - x1 * sin
    // x1' = x1 * cos + x0 * sin
    void apply_rope(
        float* vec,
        size_t position
    ) const {
#if defined(__AVX512F__)
        // Process in 16-element AVX-512 slices
        for (size_t i = 0; i < HALF_DIM; i += 8) {
            alignas(64) float cos_vals[8];
            alignas(64) float sin_vals[8];
            alignas(64) float x0[8];
            alignas(64) float x1[8];

            for (size_t j = 0; j < 8; ++j) {
                float theta = static_cast<float>(position) * inv_freq_[i + j];
                cos_vals[j] = std::cos(theta);
                sin_vals[j] = std::sin(theta);
                x0[j] = vec[(i + j) * 2 + 0];
                x1[j] = vec[(i + j) * 2 + 1];
            }

            __m256 vx0   = _mm256_loadu_ps(x0);
            __m256 vx1   = _mm256_loadu_ps(x1);
            __m256 vcos  = _mm256_loadu_ps(cos_vals);
            __m256 vsin  = _mm256_loadu_ps(sin_vals);

            // x0_out = x0 * cos - x1 * sin
            __m256 vx0_out = _mm256_fmsub_ps(vx0, vcos, _mm256_mul_ps(vx1, vsin));
            // x1_out = x1 * cos + x0 * sin
            __m256 vx1_out = _mm256_fmadd_ps(vx1, vcos, _mm256_mul_ps(vx0, vsin));

            alignas(64) float out0[8];
            alignas(64) float out1[8];
            _mm256_storeu_ps(out0, vx0_out);
            _mm256_storeu_ps(out1, vx1_out);

            for (size_t j = 0; j < 8; ++j) {
                vec[(i + j) * 2 + 0] = out0[j];
                vec[(i + j) * 2 + 1] = out1[j];
            }
        }
#else
        for (size_t i = 0; i < HALF_DIM; ++i) {
            float theta = static_cast<float>(position) * inv_freq_[i];
            float cos_th = std::cos(theta);
            float sin_th = std::sin(theta);

            float x0 = vec[i * 2 + 0];
            float x1 = vec[i * 2 + 1];

            vec[i * 2 + 0] = x0 * cos_th - x1 * sin_th;
            vec[i * 2 + 1] = x1 * cos_th + x0 * sin_th;
        }
#endif
    }

private:
    std::vector<float> inv_freq_;
};

} // namespace weft
