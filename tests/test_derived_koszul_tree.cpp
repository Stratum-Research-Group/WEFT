#include <iostream>
#include <cassert>
#include <vector>
#include "csrc/cli/weft_derived_koszul_tree.h"

int main() {
    std::cout << "[Test] Initializing Derived Koszul Tree Decoding SIMD Test..." << std::endl;

    const size_t num_nodes = 8;
    const float threshold = 0.0f;

    std::vector<float> target_scores(num_nodes, 1.0f);
    std::vector<float> draft_scores(num_nodes, 0.5f);
    std::vector<int32_t> parent_indices = { -1, 0, 0, 1, 1, 2, 2, 3 };
    std::vector<uint8_t> out_accepted(num_nodes, 0);

    // Fail node 1
    target_scores[1] = -5.0f;
    draft_scores[1] = 5.0f; // diff = -10 < 0 -> node 1 rejected

    size_t accepted = weft::avx512_verify_koszul_tree_nodes(
        target_scores.data(),
        draft_scores.data(),
        parent_indices.data(),
        out_accepted.data(),
        num_nodes,
        threshold
    );

    // Node 1 rejected -> descendants 3, 4, 7 must also be rejected
    assert(out_accepted[0] == 1);
    assert(out_accepted[1] == 0);
    assert(out_accepted[3] == 0);
    assert(out_accepted[4] == 0);
    assert(out_accepted[7] == 0);

    // Node 2 accepted -> 5, 6 accepted
    assert(out_accepted[2] == 1);
    assert(out_accepted[5] == 1);
    assert(out_accepted[6] == 1);
    assert(accepted == 4);

    std::cout << "[Test] Accepted Nodes Count: " << accepted << " / " << num_nodes << std::endl;
    std::cout << "[Test] Derived Koszul Tree Decoding SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
