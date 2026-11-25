#include <iostream>
#include <vector>
#include <cassert>
#include "csrc/cli/weft_numa_topology_tuner.h"

int main() {
    std::cout << "[TEST] Running test_numa_topology_tuner..." << std::endl;

    auto topology = weft::NUMATopologyTuner::discover_topology();
    std::cout << "  Discovered NUMA nodes: " << topology.size() << std::endl;
    assert(!topology.empty());

    for (const auto& node : topology) {
        std::cout << "    Node " << node.node_id << " core count: " << node.cpu_cores.size() << std::endl;
        assert(!node.cpu_cores.empty());
    }

    auto cores = weft::NUMATopologyTuner::get_optimal_affinity_cores(topology, 4);
    std::cout << "  Optimal core affinity for 4 workers: ";
    for (int c : cores) std::cout << c << " ";
    std::cout << std::endl;
    assert(cores.size() == 4 || cores.size() == topology[0].cpu_cores.size());

    std::cout << "[PASS] test_numa_topology_tuner succeeded." << std::endl;
    return 0;
}
