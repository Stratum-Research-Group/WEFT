#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

namespace weft {

// Co-Quantized 2-bit activation structure
struct WeftCoQuantActivation {
    float scale;
    int32_t zero_point;
    int32_t sum_x;
    std::vector<uint8_t> packed_act; // 4 2-bit values per byte
};

class WeftCoQuantGEMVEngine {
public:
    size_t in_dim;
    size_t out_dim;
    std::vector<int32_t> precomputed_row_sums; // Sum of w_ij for each row i

    WeftCoQuantGEMVEngine(size_t in_dim, size_t out_dim)
        : in_dim(in_dim), out_dim(out_dim), precomputed_row_sums(out_dim, 0) {}

    // Quantize dynamic activation vector into 2-bit packed structure
    WeftCoQuantActivation quantize_activation(const float* x) {
        WeftCoQuantActivation act;
        float x_min = x[0], x_max = x[0];
        for (size_t j = 1; j < in_dim; ++j) {
            if (x[j] < x_min) x_min = x[j];
            if (x[j] > x_max) x_max = x[j];
        }

        float s = (x_max - x_min > 1e-6f) ? (x_max - x_min) / 3.0f : 1.0f;
        int32_t z = static_cast<int32_t>(std::round(-x_min / s));
        act.scale = s;
        act.zero_point = z;
        act.sum_x = 0;

        act.packed_act.resize((in_dim + 3) / 4, 0);
        for (size_t j = 0; j < in_dim; ++j) {
            int q = std::clamp(static_cast<int>(std::round(x[j] / s + z)), 0, 3);
            act.sum_x += q;
            size_t byte_idx = j / 4;
            size_t shift = (j % 4) * 2;
            act.packed_act[byte_idx] |= (static_cast<uint8_t>(q) & 0x03) << shift;
        }

        return act;
    }

    // Co-quantized GEMV evaluation: y_i = s_w * s_x * [ sum(w*x) - z_x*sum_w - z_w*sum_x + N*z_w*z_x ]
    void compute_gemv(const WeftCoQuantActivation& act, const uint8_t* const* packed_weights,
                      float weight_scale, int32_t weight_zero, float* y) {
        float combined_scale = act.scale * weight_scale;
        int32_t zw_zx_N = static_cast<int32_t>(in_dim) * weight_zero * act.zero_point;
        int32_t zw_sum_x = weight_zero * act.sum_x;

        for (size_t i = 0; i < out_dim; ++i) {
            int32_t integer_dot = 0;
            const uint8_t* w_row = packed_weights[i];
            size_t num_bytes = (in_dim + 3) / 4;

            for (size_t b = 0; b < num_bytes; ++b) {
                uint8_t w_byte = w_row[b];
                uint8_t x_byte = act.packed_act[b];
                // Unpack 4 2-bit values
                for (size_t k = 0; k < 4 && (b * 4 + k) < in_dim; ++k) {
                    uint8_t w_val = (w_byte >> (k * 2)) & 0x03;
                    uint8_t x_val = (x_byte >> (k * 2)) & 0x03;
                    integer_dot += static_cast<int32_t>(w_val) * static_cast<int32_t>(x_val);
                }
            }

            int32_t zx_sum_w = act.zero_point * precomputed_row_sums[i];
            int32_t compensated_dot = integer_dot - zx_sum_w - zw_sum_x + zw_zx_N;
            y[i] = static_cast<float>(compensated_dot) * combined_scale;
        }
    }
};

} // namespace weft
