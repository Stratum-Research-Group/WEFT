#pragma once
// weft_multimodal_kv.h - Asymmetric Multi-Modal KV-Cache Quantization & Cross-Attention
// Milestone Paper #128
// Authors: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <immintrin.h>

namespace weft {

enum class ModalityType : uint8_t {
    TEXT = 0,   // Standard 2-bit / 4-bit uniform asymmetric quantization
    VISION = 1  // Ultra-compact 1.5-bit ternary quantization {-1, 0, +1}
};

struct MultiModalTokenBlock {
    static constexpr size_t DIM = 64;

    ModalityType modality{ModalityType::TEXT};
    float scale{1.0f};
    float zero_point{0.0f};

    // Bitmasks for ternary representation (ModalityType::VISION):
    // pos_mask: bit d = 1 if coordinate is +1
    // neg_mask: bit d = 1 if coordinate is -1
    // (coordinate is 0 if neither bit is set)
    uint64_t pos_mask{0};
    uint64_t neg_mask{0};

    // 2-bit packed data for text tokens (ModalityType::TEXT)
    uint8_t text_packed[DIM * 2 / 8];
};

class WeftMultiModalKVEngine {
public:
    WeftMultiModalKVEngine() = default;

    // Fused dot product between FP32 query and heterogeneous multimodal token block
    static float compute_token_dot_product(
        const float* query,
        const MultiModalTokenBlock& block
    ) {
        float dot = 0.0f;

        if (block.modality == ModalityType::VISION) {
            // Visual Token: Ternary dot-product using bitwise masking (no integer muls)
            float pos_sum = 0.0f;
            float neg_sum = 0.0f;

            for (size_t d = 0; d < MultiModalTokenBlock::DIM; ++d) {
                if (block.pos_mask & (1ULL << d)) {
                    pos_sum += query[d];
                } else if (block.neg_mask & (1ULL << d)) {
                    neg_sum += query[d];
                }
            }

            dot = (pos_sum - neg_sum) * block.scale;
        } else {
            // Textual Token: 2-bit dequantization dot-product
            for (size_t d = 0; d < MultiModalTokenBlock::DIM; ++d) {
                size_t byte_idx = (d * 2) / 8;
                size_t shift = (d * 2) % 8;
                uint8_t q_val = (block.text_packed[byte_idx] >> shift) & 0x03;
                float k_val = static_cast<float>(q_val) * block.scale + block.zero_point;
                dot += query[d] * k_val;
            }
        }

        return dot / std::sqrt(static_cast<float>(MultiModalTokenBlock::DIM));
    }

    // Compute cross-attention over heterogeneous multimodal tokens
    void compute_cross_attention(
        const float* query,
        const MultiModalTokenBlock* blocks,
        size_t total_blocks,
        float* out_scores
    ) {
        for (size_t i = 0; i < total_blocks; ++i) {
            out_scores[i] = compute_token_dot_product(query, blocks[i]);
        }
    }
};

} // namespace weft
