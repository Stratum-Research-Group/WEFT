#pragma once
// weft_rope_context_extender.h - Attention Sink Preservation & RoPE Context Extender
// Dynamic NTK-aware frequency scaling for long-context speculative verification (8k -> 64k tokens).

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>

namespace weft {

class WeftRoPEContextExtender {
public:
    float base_freq;
    size_t orig_context_len;
    size_t target_context_len;
    float scaling_factor;

    WeftRoPEContextExtender(float base_freq = 10000.0f, size_t orig_context_len = 8192, size_t target_context_len = 65536)
        : base_freq(base_freq), orig_context_len(orig_context_len), target_context_len(target_context_len) {
        scaling_factor = static_cast<float>(target_context_len) / static_cast<float>(orig_context_len);
    }

    // Compute NTK-scaled inverse frequency for dimension index d in [0, head_dim/2)
    float get_scaled_inv_freq(size_t d, size_t head_dim, size_t current_seq_len) const {
        float effective_scale = 1.0f;
        if (current_seq_len > orig_context_len) {
            float ratio = static_cast<float>(current_seq_len) / static_cast<float>(orig_context_len);
            effective_scale = std::min(scaling_factor, ratio);
        }

        // Dynamic NTK base adjustment: base_adj = base * (scale ^ (dim / (dim - 2)))
        float exponent = static_cast<float>(head_dim) / static_cast<float>(head_dim - 2);
        float adj_base = base_freq * std::pow(effective_scale, exponent);

        return 1.0f / std::pow(adj_base, static_cast<float>(2 * d) / static_cast<float>(head_dim));
    }

    // Apply rotary embedding in-place to 2D vector pair [q0, q1] at position m
    void apply_rotary_pair(float& q0, float& q1, size_t m, float inv_freq) const {
        float theta = static_cast<float>(m) * inv_freq;
        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);

        float r0 = q0 * cos_theta - q1 * sin_theta;
        float r1 = q0 * sin_theta + q1 * cos_theta;
        q0 = r0;
        q1 = r1;
    }
};

} // namespace weft
