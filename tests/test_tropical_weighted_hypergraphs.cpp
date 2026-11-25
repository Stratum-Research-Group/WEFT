#include "csrc/cli/weft_tropical_weighted_hypergraphs.h"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    float raw_logits[8] = {2.0f, 2.0f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    // Expert 0 has heavy load (100 tokens), Expert 1 has light load (0 tokens)
    uint32_t expert_loads[8] = {100, 0, 0, 0, 0, 0, 0, 0};
    uint32_t top2_indices[2] = {0, 0};
    float top2_weights[2] = {0.0f, 0.0f};

    weft::tropical_moe_route_8experts_avx512(
        raw_logits, expert_loads, top2_indices, top2_weights, 0.05f);

    // Expert 1 should be selected over Expert 0 due to penalty
    assert(top2_indices[0] == 1);
    assert(top2_weights[0] > 0.5f);

    std::cout << "[PASS] Tropical Weighted Hypergraphs MoE Router Unit Test passed (Top-1 Expert: " 
              << top2_indices[0] << ", weight: " << top2_weights[0] << ").\n";
    return 0;
}
