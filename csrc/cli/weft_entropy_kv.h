#pragma once
// weft_entropy_kv.h - Entropy-Guided Dynamic KV Compaction & Precision Allocation
// Milestone Paper #136
// Authors: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

enum class EntropyPrecisionTier : uint8_t {
    HIGH_4BIT = 0,   // High-entropy branching tokens (4-bit asymmetric)
    MED_2BIT  = 1,   // Medium-entropy context tokens (2-bit asymmetric)
    LOW_TERNARY = 2  // Low-entropy predictable boilerplate (1.5-bit ternary)
};

struct EntropyKVBlock {
    static constexpr size_t DIM = 64;

    uint32_t token_id{0};
    EntropyPrecisionTier tier{EntropyPrecisionTier::MED_2BIT};
    float scale{1.0f};
    float zero_point{0.0f};
    float entropy{0.0f};

    // Modality bitmasks for ternary representation (LOW_TERNARY):
    uint64_t pos_mask{0};
    uint64_t neg_mask{0};

    // Packed 2-bit or 4-bit storage:
    // 4-bit: 32 bytes | 2-bit: 16 bytes
    alignas(64) uint8_t packed_data[32];
};

class WeftEntropyKVEngine {
public:
    static constexpr size_t TOP_K = 8;

    WeftEntropyKVEngine(float tau_high = 1.8f, float tau_low = 0.5f)
        : tau_high_(tau_high), tau_low_(tau_low) {}

    // In-register computation of Shannon entropy surrogate from top-K unnormalized logits
    static float compute_entropy_surrogate(const float* top_k_logits, size_t k = TOP_K) {
        float max_l = -1e9f;
        for (size_t i = 0; i < k; ++i) {
            if (top_k_logits[i] > max_l) max_l = top_k_logits[i];
        }

        float sum_exp = 0.0f;
        float exp_vals[TOP_K];
        for (size_t i = 0; i < k; ++i) {
            exp_vals[i] = std::exp(top_k_logits[i] - max_l);
            sum_exp += exp_vals[i];
        }

        float entropy = 0.0f;
        float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (size_t i = 0; i < k; ++i) {
            float p = exp_vals[i] * inv_sum;
            if (p > 1e-6f) {
                entropy -= p * std::log2(p);
            }
        }

        return entropy;
    }

    // Determine target precision tier based on entropy
    EntropyPrecisionTier classify_tier(float entropy) const {
        if (entropy >= tau_high_) return EntropyPrecisionTier::HIGH_4BIT;
        if (entropy >= tau_low_)  return EntropyPrecisionTier::MED_2BIT;
        return EntropyPrecisionTier::LOW_TERNARY;
    }

    // Compute dot product between query and entropy-quantized token block
    static float compute_token_dot_product(
        const float* query,
        const EntropyKVBlock& block
    ) {
        float dot = 0.0f;

        if (block.tier == EntropyPrecisionTier::LOW_TERNARY) {
            // Ternary fast addition/subtraction
            float pos_sum = 0.0f;
            float neg_sum = 0.0f;
            for (size_t d = 0; d < EntropyKVBlock::DIM; ++d) {
                if (block.pos_mask & (1ULL << d)) pos_sum += query[d];
                else if (block.neg_mask & (1ULL << d)) neg_sum += query[d];
            }
            dot = (pos_sum - neg_sum) * block.scale;
        } else if (block.tier == EntropyPrecisionTier::MED_2BIT) {
            // 2-bit unpacked dot product
            for (size_t d = 0; d < EntropyKVBlock::DIM; ++d) {
                size_t byte_idx = (d * 2) / 8;
                size_t shift = (d * 2) % 8;
                uint8_t q = (block.packed_data[byte_idx] >> shift) & 0x03;
                float k_val = static_cast<float>(q) * block.scale + block.zero_point;
                dot += query[d] * k_val;
            }
        } else {
            // 4-bit unpacked dot product
            for (size_t d = 0; d < EntropyKVBlock::DIM; ++d) {
                size_t byte_idx = d / 2;
                size_t shift = (d % 2) * 4;
                uint8_t q = (block.packed_data[byte_idx] >> shift) & 0x0F;
                float k_val = static_cast<float>(q) * block.scale + block.zero_point;
                dot += query[d] * k_val;
            }
        }

        return dot / std::sqrt(static_cast<float>(EntropyKVBlock::DIM));
    }

private:
    float tau_high_;
    float tau_low_;
};

} // namespace weft
