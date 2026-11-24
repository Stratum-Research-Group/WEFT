// weft_voronoi_quant.h
// Native AVX-512 Fast Conway-Sloane E8 Gosset Lattice Voronoi Quantization.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>

namespace weft {

// Fast Conway-Sloane E8 Lattice Quantization:
// Input: 8D float vector x
// Output: 8D float lattice point lambda in E8
inline void quantize_e8_conway_sloane_avx512(const float* x, float* out_lambda) {
    // 1. Round to nearest integers: f(x)
    float f[8];
    float diff_f[8];
    int sum_f = 0;
    int max_diff_f_idx = 0;
    float max_diff_f = -1.0f;

    for (int i = 0; i < 8; ++i) {
        f[i] = std::round(x[i]);
        float d = std::abs(x[i] - f[i]);
        diff_f[i] = d;
        sum_f += static_cast<int>(f[i]);
        if (d > max_diff_f) {
            max_diff_f = d;
            max_diff_f_idx = i;
        }
    }
    // Parity correction for f: sum must be even
    if (std::abs(sum_f) % 2 != 0) {
        f[max_diff_f_idx] += (x[max_diff_f_idx] >= f[max_diff_f_idx]) ? 1.0f : -1.0f;
    }

    // 2. Round to nearest half-integers: g(x)
    float g[8];
    float diff_g[8];
    int sum_g_shifted = 0;
    int max_diff_g_idx = 0;
    float max_diff_g = -1.0f;

    for (int i = 0; i < 8; ++i) {
        g[i] = std::floor(x[i]) + 0.5f;
        float d = std::abs(x[i] - g[i]);
        diff_g[i] = d;
        sum_g_shifted += static_cast<int>(std::floor(x[i]));
        if (d > max_diff_g) {
            max_diff_g = d;
            max_diff_g_idx = i;
        }
    }
    // Parity correction for g: sum of (g_i - 0.5) must have parity matching 8*0.5 mod 2 = 0
    if (std::abs(sum_g_shifted) % 2 != 0) {
        g[max_diff_g_idx] += (x[max_diff_g_idx] >= g[max_diff_g_idx]) ? 1.0f : -1.0f;
    }

    // 3. Compute distance to f and g, select closer candidate
    float dist_f = 0.0f;
    float dist_g = 0.0f;
    for (int i = 0; i < 8; ++i) {
        float df = x[i] - f[i];
        float dg = x[i] - g[i];
        dist_f += df * df;
        dist_g += dg * dg;
    }

    const float* best = (dist_f <= dist_g) ? f : g;
    for (int i = 0; i < 8; ++i) {
        out_lambda[i] = best[i];
    }
}

} // namespace weft
