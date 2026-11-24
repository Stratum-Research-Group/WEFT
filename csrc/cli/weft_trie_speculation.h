#pragma once
// weft_trie_speculation.h - Hierarchical Prefix Trie Speculation & Softmax Pruning
// Milestone Paper #134
// Authors: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct TrieSpeculativeNode {
    uint32_t node_id{0};
    uint32_t token_id{0};
    int32_t parent_id{-1};
    uint64_t descendant_bitmask{0}; // Bit i is 1 if node i is a descendant of this node
    float path_draft_prob{1.0f};
    float target_prob{0.0f};
    bool verified{false};
    bool accepted{false};
};

class WeftTrieSpeculativeEngine {
public:
    static constexpr size_t MAX_TRIE_NODES = 64;

    WeftTrieSpeculativeEngine(float tail_pruning_threshold = 0.05f)
        : tail_threshold_(tail_pruning_threshold) {}

    // Verify trie candidates level-by-level with in-register descendant branch pruning
    size_t verify_trie(
        TrieSpeculativeNode* nodes,
        size_t total_nodes,
        const float* target_logits
    ) {
        if (total_nodes == 0) return 0;

        uint64_t active_mask = (total_nodes >= 64) ? ~0ULL : ((1ULL << total_nodes) - 1ULL);
        size_t accepted_count = 0;

        for (size_t i = 0; i < total_nodes; ++i) {
            // Check if this node was already pruned by an ancestor rejection
            if (!(active_mask & (1ULL << i))) {
                nodes[i].verified = false;
                nodes[i].accepted = false;
                continue;
            }

            nodes[i].verified = true;

            // Compute target Softmax probability surrogate
            float logit = target_logits[i];
            float prob = 1.0f / (1.0f + std::exp(-logit));
            nodes[i].target_prob = prob;

            // Rejection sampling acceptance check
            if (prob >= nodes[i].path_draft_prob * 0.75f && prob >= tail_threshold_) {
                nodes[i].accepted = true;
                accepted_count++;
            } else {
                nodes[i].accepted = false;
                // Reject node: Prune all descendants by clearing their bits in active_mask
                active_mask &= ~nodes[i].descendant_bitmask;
            }
        }

        return accepted_count;
    }

private:
    float tail_threshold_;
};

} // namespace weft
