#pragma once
// weft_dynamic_adapter.h - Multi-Model Hot-Swapping & Dynamic Linear Adapter Blending
// Supports zero-downtime hot-swapping and weighted linear blending: W_eff = W_base + alpha * A1*B1 + beta * A2*B2

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <cstring>

namespace weft {

struct LoRAWeightPair {
    std::string name;
    size_t in_dim;
    size_t out_dim;
    size_t rank;
    std::vector<float> lora_a; // rank x in_dim
    std::vector<float> lora_b; // out_dim x rank
    float scale{1.0f};
};

class WeftDynamicAdapterBlender {
public:
    WeftDynamicAdapterBlender(size_t in_dim, size_t out_dim, size_t max_rank = 16)
        : in_dim_(in_dim), out_dim_(out_dim), max_rank_(max_rank) {}

    // Register or hot-swap an active adapter
    void register_adapter(const std::string& name, size_t rank, float scale = 1.0f) {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        LoRAWeightPair pair;
        pair.name = name;
        pair.in_dim = in_dim_;
        pair.out_dim = out_dim_;
        pair.rank = rank;
        pair.scale = scale;
        pair.lora_a.resize(rank * in_dim_, 0.01f);
        pair.lora_b.resize(out_dim_ * rank, 0.01f);
        adapters_[name] = std::move(pair);
    }

    // Unload adapter
    void unload_adapter(const std::string& name) {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        adapters_.erase(name);
    }

    // Compute blended forward adapter delta: y_delta = sum_i(weight_i * scale_i * B_i * (A_i * x))
    void blend_and_apply(const float* x, const std::vector<std::pair<std::string, float>>& blend_spec, float* out_delta) {
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        std::memset(out_delta, 0, sizeof(float) * out_dim_);

        std::vector<float> intermediate(max_rank_, 0.0f);

        for (const auto& item : blend_spec) {
            const auto& name = item.first;
            float blend_w = item.second;
            auto it = adapters_.find(name);
            if (it == adapters_.end()) continue;

            const auto& ad = it->second;
            float total_scale = blend_w * ad.scale;

            // 1. intermediate = A * x (rank x 1)
            for (size_t r = 0; r < ad.rank; ++r) {
                float sum = 0.0f;
                for (size_t i = 0; i < in_dim_; ++i) {
                    sum += ad.lora_a[r * in_dim_ + i] * x[i];
                }
                intermediate[r] = sum;
            }

            // 2. out_delta += total_scale * B * intermediate (out_dim x 1)
            for (size_t o = 0; o < out_dim_; ++o) {
                float sum = 0.0f;
                for (size_t r = 0; r < ad.rank; ++r) {
                    sum += ad.lora_b[o * ad.rank + r] * intermediate[r];
                }
                out_delta[o] += total_scale * sum;
            }
        }
    }

private:
    size_t in_dim_;
    size_t out_dim_;
    size_t max_rank_;
    std::unordered_map<std::string, LoRAWeightPair> adapters_;
    std::shared_mutex rw_mutex_;
};

} // namespace weft
