#include "csrc/cli/weft_tropical_metric_trees.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>

int main() {
    std::vector<float> cost(256, 1000.0f); // High initial distance
    std::vector<float> dist(256, 0.0f);
    std::vector<uint8_t> next_hop(256, 0);

    for (size_t i = 0; i < 16; ++i) {
        cost[i * 16 + i] = 0.0f;
    }

    // Set up a path: Node 0 -> Node 1 (latency 2.0), Node 1 -> Node 2 (latency 3.0)
    // Direct Node 0 -> Node 2 is 10.0 (suboptimal)
    cost[0 * 16 + 1] = 2.0f;
    cost[1 * 16 + 2] = 3.0f;
    cost[0 * 16 + 2] = 10.0f;

    weft::tropical_metric_trees_route_16nodes_avx512(cost.data(), dist.data(), next_hop.data());

    // Geodesic distance 0 -> 2 should be 2.0 + 3.0 = 5.0
    assert(dist[0 * 16 + 2] == 5.0f);
    // Next hop from 0 to 2 should be Node 1
    assert(next_hop[0 * 16 + 2] == 1);

    std::cout << "[PASS] Tropical Metric Trees Routing Unit Test passed (Geodesic 0->2: " 
              << dist[0 * 16 + 2] << ", next_hop: " << (int)next_hop[0 * 16 + 2] << ").\n";
    return 0;
}
