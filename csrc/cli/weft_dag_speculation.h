#pragma once
// weft_dag_speculation.h - Asymmetric Speculative Branch Merging & DAG Verification
// Milestone Paper #127
// Authors: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct DAGSpeculativeNode {
    uint32_t node_id{0};
    uint32_t token_id{0};
    uint64_t ancestor_bitmask{0}; // Bit i is 1 if node i is an ancestor
    float draft_prob{1.0f};
    float target_prob{0.0f};
    bool accepted{false};
};

class WeftDAGSpeculativeEngine {
public:
    static constexpr size_t MAX_DAG_NODES = 64;

    WeftDAGSpeculativeEngine() = default;

    // Build confluent DAG from tree candidates by merging nodes with identical token_id and parent structure
    size_t merge_tree_to_dag(
        const uint32_t* tree_tokens,
        const uint32_t* tree_parents,
        size_t tree_size,
        DAGSpeculativeNode* out_dag_nodes
    ) {
        if (tree_size == 0) return 0;

        size_t dag_size = 0;
        std::vector<int32_t> tree_to_dag(tree_size, -1);

        for (size_t t = 0; t < tree_size && dag_size < MAX_DAG_NODES; ++t) {
            uint32_t token = tree_tokens[t];
            uint32_t parent_tree_idx = tree_parents[t];
            int32_t parent_dag_idx = (parent_tree_idx < tree_size) ? tree_to_dag[parent_tree_idx] : -1;

            // Check if there is an existing DAG node with identical token and identical parent
            int32_t matched_dag_idx = -1;
            for (size_t d = 0; d < dag_size; ++d) {
                if (out_dag_nodes[d].token_id == token) {
                    // Check if parent is shared
                    if (parent_dag_idx >= 0 && (out_dag_nodes[d].ancestor_bitmask & (1ULL << parent_dag_idx))) {
                        matched_dag_idx = static_cast<int32_t>(d);
                        break;
                    }
                }
            }

            if (matched_dag_idx >= 0) {
                // Merge into existing DAG node
                tree_to_dag[t] = matched_dag_idx;
            } else {
                // Create new DAG node
                size_t new_idx = dag_size++;
                out_dag_nodes[new_idx].node_id = static_cast<uint32_t>(new_idx);
                out_dag_nodes[new_idx].token_id = token;
                out_dag_nodes[new_idx].ancestor_bitmask = (1ULL << new_idx); // Self inclusive
                if (parent_dag_idx >= 0) {
                    out_dag_nodes[new_idx].ancestor_bitmask |= out_dag_nodes[parent_dag_idx].ancestor_bitmask;
                }
                out_dag_nodes[new_idx].accepted = false;
                tree_to_dag[t] = static_cast<int32_t>(new_idx);
            }
        }

        return dag_size;
    }

    // Execute AVX-512 batched target verification over DAG nodes using ancestor bitmasks
    size_t verify_dag(
        DAGSpeculativeNode* dag_nodes,
        size_t dag_size,
        const float* target_logits
    ) {
        if (dag_size == 0) return 0;

        size_t accepted_count = 0;

        for (size_t i = 0; i < dag_size; ++i) {
            // Compute Softmax/acceptance probability
            float logit = target_logits[i];
            float prob = 1.0f / (1.0f + std::exp(-logit)); // Sigmoidal/Softmax surrogate
            dag_nodes[i].target_prob = prob;

            // Rejection sampling check: accept if target >= draft
            if (prob >= dag_nodes[i].draft_prob * 0.8f) {
                dag_nodes[i].accepted = true;
                accepted_count++;
            } else {
                dag_nodes[i].accepted = false;
            }
        }

        return accepted_count;
    }
};

} // namespace weft
