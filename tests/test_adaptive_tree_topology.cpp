#include <iostream>
#include <cassert>
#include <vector>
#include "../csrc/cli/weft_adaptive_tree.h"

int main() {
    std::cout << "[TEST] Running Weft Adaptive Tree Topology Verification..." << std::endl;

    weft::WeftAdaptiveTreeBuilder builder(16);

    // 1. Low Entropy Test (Code / Math): Should produce 8x1 linear chain
    auto topo_low = builder.build_topology(0.2f, 1.0f);
    assert(topo_low.depth == 8);
    assert(topo_low.breadth == 1);
    assert(topo_low.total_nodes == 8);
    assert(topo_low.parent_indices[0] == -1);
    assert(topo_low.parent_indices[1] == 0);
    assert(topo_low.parent_indices[7] == 6);
    std::cout << "[PASS] Low-entropy deep linear chain verified." << std::endl;

    // 2. High Entropy Test (Creative): Should produce 2x8 broad bush
    auto topo_high = builder.build_topology(2.5f, 1.0f);
    assert(topo_high.depth == 2);
    assert(topo_high.breadth == 8);
    assert(topo_low.total_nodes <= 16);
    std::cout << "[PASS] High-entropy broad bush verified." << std::endl;

    std::cout << "[ALL PASSED] Adaptive Tree Topology tests successful." << std::endl;
    return 0;
}
