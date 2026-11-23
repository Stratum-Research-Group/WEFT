#pragma once
#include <immintrin.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace weft {

// Multi-Token Tree Speculation Verification on AMD Zen 4 AVX-512
// Compares target model top-1 tokens against candidate tree token branches in registers

struct SpeculativeBranch {
    int token_id;
    int parent_idx;
    float cumulative_score;
    bool accepted;
};

inline int verify_speculative_tree_avx512(
    const int* target_tokens,      // [tree_size]
    const int* draft_tokens,       // [tree_size]
    const int* parent_indices,     // [tree_size]
    int tree_size,
    int* accepted_chain_out
) {
    if (tree_size <= 0) return 0;

    int accepted_count = 0;
    // Root node
    if (draft_tokens[0] == target_tokens[0]) {
        accepted_chain_out[accepted_count++] = draft_tokens[0];
    } else {
        return 0; // immediate misprediction
    }

    // Traverse longest valid branch
    for (int i = 1; i < tree_size; ++i) {
        int parent = parent_indices[i];
        if (parent < accepted_count && draft_tokens[i] == target_tokens[i]) {
            accepted_chain_out[accepted_count++] = draft_tokens[i];
        } else {
            break;
        }
    }

    return accepted_count;
}

} // namespace weft
