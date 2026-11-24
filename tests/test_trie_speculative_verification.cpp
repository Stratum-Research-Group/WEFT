// test_trie_speculative_verification.cpp - Verification for Paper #134
#include "csrc/cli/weft_trie_speculation.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace weft;

int main() {
    std::cout << "[TEST] Initializing test_trie_speculative_verification..." << std::endl;

    // Construct a 7-node candidate trie:
    // Node 0: root (token 10) -> parent -1, descendants {1, 2, 3, 4, 5, 6}
    // Node 1: child of 0 (token 20) -> descendants {3, 4}
    // Node 2: child of 0 (token 21) -> descendants {5, 6}
    // Node 3: child of 1 (token 30) -> leaf
    // Node 4: child of 1 (token 31) -> leaf
    // Node 5: child of 2 (token 32) -> leaf
    // Node 6: child of 2 (token 33) -> leaf
    constexpr size_t TOTAL_NODES = 7;
    std::vector<TrieSpeculativeNode> nodes(TOTAL_NODES);

    for (size_t i = 0; i < TOTAL_NODES; ++i) {
        nodes[i].node_id = static_cast<uint32_t>(i);
        nodes[i].path_draft_prob = 0.8f;
        nodes[i].verified = false;
        nodes[i].accepted = false;
    }

    nodes[0].token_id = 10; nodes[0].parent_id = -1;
    nodes[0].descendant_bitmask = (1ULL << 1) | (1ULL << 2) | (1ULL << 3) | (1ULL << 4) | (1ULL << 5) | (1ULL << 6);

    nodes[1].token_id = 20; nodes[1].parent_id = 0;
    nodes[1].descendant_bitmask = (1ULL << 3) | (1ULL << 4);

    nodes[2].token_id = 21; nodes[2].parent_id = 0;
    nodes[2].descendant_bitmask = (1ULL << 5) | (1ULL << 6);

    nodes[3].token_id = 30; nodes[3].parent_id = 1; nodes[3].descendant_bitmask = 0;
    nodes[4].token_id = 31; nodes[4].parent_id = 1; nodes[4].descendant_bitmask = 0;
    nodes[5].token_id = 32; nodes[5].parent_id = 2; nodes[5].descendant_bitmask = 0;
    nodes[6].token_id = 33; nodes[6].parent_id = 2; nodes[6].descendant_bitmask = 0;

    // Simulate target logits:
    // Root (0) accepted (logit = 2.0 -> prob ~0.88)
    // Child 1 rejected (logit = -2.0 -> prob ~0.11 < 0.6)
    // Child 2 accepted (logit = 1.5 -> prob ~0.81)
    // Because Child 1 is rejected, nodes 3 and 4 MUST be pruned without verification!
    // Nodes 5 and 6 should be verified.
    std::vector<float> target_logits = {2.0f, -2.0f, 1.5f, 2.0f, 2.0f, 1.2f, -1.0f};

    WeftTrieSpeculativeEngine engine(/*tail_pruning_threshold=*/0.05f);
    size_t accepted = engine.verify_trie(nodes.data(), TOTAL_NODES, target_logits.data());

    std::cout << "[TEST] Accepted Nodes: " << accepted << " / " << TOTAL_NODES << std::endl;

    // Node 0 accepted
    assert(nodes[0].verified && nodes[0].accepted);
    // Node 1 verified but rejected
    assert(nodes[1].verified && !nodes[1].accepted);
    // Node 2 verified and accepted
    assert(nodes[2].verified && nodes[2].accepted);

    // Nodes 3 and 4 were under rejected Node 1 -> MUST NOT be verified!
    assert(!nodes[3].verified && !nodes[3].accepted);
    assert(!nodes[4].verified && !nodes[4].accepted);
    std::cout << "[TEST] Nodes 3 & 4 successfully bypassed verification due to ancestor rejection!" << std::endl;

    // Nodes 5 and 6 were under accepted Node 2 -> MUST be verified!
    assert(nodes[5].verified && nodes[5].accepted);
    assert(nodes[6].verified && !nodes[6].accepted); // target_logit = -1.0 rejected

    std::cout << "[TEST] All hierarchical prefix trie speculation checks passed successfully!" << std::endl;
    return 0;
}
