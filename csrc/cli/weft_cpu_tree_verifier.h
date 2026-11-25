#pragma once

#include "weft_adaptive_tree.h"
#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <omp.h>
#include <algorithm>

namespace weft {

struct TreeVerificationResult {
    std::vector<int32_t> accepted_tokens;
    size_t accepted_count;
    int32_t best_leaf;
};

class CPUSpeculativeTreeVerifier {
public:
    static inline void build_causal_tree_mask_avx512(
        const int32_t* parent_indices,
        float* mask,
        size_t num_nodes
    ) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < num_nodes; ++i) {
            float* row = mask + (i * num_nodes);
            for (size_t j = 0; j < num_nodes; ++j) {
                row[j] = -1e9f;
            }
            row[i] = 0.0f; // Self-attention

            int32_t curr = parent_indices[i];
            while (curr != -1 && curr >= 0 && static_cast<size_t>(curr) < num_nodes) {
                row[curr] = 0.0f;
                curr = parent_indices[curr];
            }
        }
    }

    static inline TreeVerificationResult verify_candidate_tree(
        const WeftTreeTopology& topo,
        const std::vector<int32_t>& draft_tokens,
        const std::vector<float>& draft_probs,
        const std::vector<float>& target_probs,
        float threshold = 0.65f
    ) {
        TreeVerificationResult res;
        res.accepted_count = 0;
        res.best_leaf = -1;

        if (draft_tokens.empty()) return res;

        // Trace down the tree finding the deepest branch where all ancestor nodes satisfy acceptance
        std::vector<bool> node_valid(topo.total_nodes, false);

        for (size_t i = 0; i < topo.total_nodes; ++i) {
            int32_t p = topo.parent_indices[i];
            bool parent_ok = (p == -1) || (static_cast<size_t>(p) < node_valid.size() && node_valid[p]);

            if (parent_ok) {
                float ratio = (i < target_probs.size() && i < draft_probs.size() && draft_probs[i] > 1e-6f)
                                  ? (target_probs[i] / draft_probs[i])
                                  : 1.0f;
                if (ratio >= threshold) {
                    node_valid[i] = true;
                    if (res.accepted_count < topo.depth) {
                        res.accepted_tokens.push_back(draft_tokens[i]);
                        res.accepted_count++;
                        res.best_leaf = static_cast<int32_t>(i);
                    }
                }
            }
        }

        if (res.accepted_tokens.empty() && !draft_tokens.empty()) {
            res.accepted_tokens.push_back(draft_tokens[0]);
            res.accepted_count = 1;
        }

        return res;
    }
};

} // namespace weft
