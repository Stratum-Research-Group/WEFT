#pragma once
// weft_dynamic_depth_scaling.h - Asynchronous Dynamic Speculative Tree Scaling Heuristic
// Dynamically tunes speculative tree depth based on rolling token acceptance rate alpha.

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cmath>

namespace weft {

class WeftDynamicDepthScaler {
public:
    size_t min_depth;
    size_t max_depth;
    size_t current_depth;
    float alpha_ema; // Exponential moving average of acceptance rate
    float smoothing;

    WeftDynamicDepthScaler(size_t min_depth = 2, size_t max_depth = 16, float initial_alpha = 0.75f)
        : min_depth(min_depth), max_depth(max_depth), current_depth(4), alpha_ema(initial_alpha), smoothing(0.15f) {}

    // Record verified tree step outcome: accepted_count / total_speculated
    size_t update_acceptance_rate(size_t accepted_count, size_t total_speculated) {
        if (total_speculated == 0) return current_depth;

        float step_alpha = static_cast<float>(accepted_count) / static_cast<float>(total_speculated);
        alpha_ema = (smoothing * step_alpha) + ((1.0f - smoothing) * alpha_ema);

        if (alpha_ema > 0.85f) {
            // High acceptance regime: greedily expand depth up to max_depth
            current_depth = std::min(max_depth, current_depth + 1);
        } else if (alpha_ema < 0.40f) {
            // Low acceptance regime: contract depth down to min_depth to avoid wasted verifier compute
            current_depth = (current_depth > min_depth + 1) ? current_depth - 2 : min_depth;
        }

        return current_depth;
    }

    size_t get_current_depth() const {
        return current_depth;
    }

    float get_alpha_ema() const {
        return alpha_ema;
    }
};

} // namespace weft
