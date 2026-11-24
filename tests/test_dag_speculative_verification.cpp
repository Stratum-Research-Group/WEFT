// test_dag_speculative_verification.cpp - Verification for Paper #127
#include "csrc/cli/weft_dag_speculation.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace weft;

int main() {
    std::cout << "[TEST] Initializing test_dag_speculative_verification..." << std::endl;

    // Define a speculative candidate tree with a redundant branch:
    // Node 0: root (token 101)
    // Node 1: child of 0 (token 202) -> Path A: 101 -> 202 -> 303 -> 404
    // Node 2: child of 1 (token 303)
    // Node 3: child of 2 (token 404)
    // Node 4: child of 0 (token 202) -> Path B: 101 -> 202 -> 303 -> 505 (shares 202 & 303)
    // Node 5: child of 4 (token 303)
    // Node 6: child of 5 (token 505)
    constexpr size_t TREE_SIZE = 7;
    std::vector<uint32_t> tree_tokens  = {101, 202, 303, 404, 202, 303, 505};
    std::vector<uint32_t> tree_parents = {  0,   0,   1,   2,   0,   4,   5};

    WeftDAGSpeculativeEngine engine;
    std::vector<DAGSpeculativeNode> dag_nodes(WeftDAGSpeculativeEngine::MAX_DAG_NODES);

    size_t dag_size = engine.merge_tree_to_dag(
        tree_tokens.data(),
        tree_parents.data(),
        TREE_SIZE,
        dag_nodes.data()
    );

    std::cout << "[TEST] Tree Size: " << TREE_SIZE << " nodes -> Merged DAG Size: " << dag_size << " nodes" << std::endl;
    // Nodes 4 and 5 should have been coalesced into nodes 1 and 2
    assert(dag_size < TREE_SIZE);
    assert(dag_size == 5); // 101, 202, 303, 404, 505

    // Verify ancestor bitmasks
    // Root (index 0) ancestor mask should be 0x1
    assert(dag_nodes[0].ancestor_bitmask == 0x1);

    // Node 505 (index 4 in DAG) should have ancestors 0, 1, 2, 4
    uint64_t expected_mask_505 = (1ULL << 0) | (1ULL << 1) | (1ULL << 2) | (1ULL << 4);
    assert(dag_nodes[4].ancestor_bitmask == expected_mask_505);

    // Verify DAG execution
    std::vector<float> target_logits = {2.0f, 1.5f, 1.2f, 0.8f, 1.1f};
    size_t accepted = engine.verify_dag(dag_nodes.data(), dag_size, target_logits.data());
    std::cout << "[TEST] Accepted DAG Nodes: " << accepted << " / " << dag_size << std::endl;
    assert(accepted > 0);

    std::cout << "[TEST] Speculative branch merging and DAG verification verified!" << std::endl;
    return 0;
}
