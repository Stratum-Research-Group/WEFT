#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "csrc/cli/weft_tropical_metric_networks.h"

int main() {
    std::cout << "[TEST] Running test_tropical_metric_networks (Paper 310)..." << std::endl;

    const size_t n = 16;
    std::vector<float> adj(n * n, 1000.0f); // Default disconnected high cost
    std::vector<float> dist(n * n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        adj[i * n + i] = 0.0f; // Self-loop distance = 0
    }

    // Linear chain: 0 -> 1 -> 2 -> ... -> 15 with cost 2.0 each
    for (size_t i = 0; i + 1 < n; ++i) {
        adj[i * n + (i + 1)] = 2.0f;
        adj[(i + 1) * n + i] = 2.0f;
    }
    // Add a shortcut from 0 to 4 with cost 3.0
    adj[0 * n + 4] = 3.0f;
    adj[4 * n + 0] = 3.0f;

    weft::TropicalMetricRouter::compute_tropical_closure(adj.data(), dist.data(), n);

    std::cout << "  Calculated shortest distance 0 -> 4: " << dist[0 * n + 4] << " (expected 3.0)" << std::endl;
    assert(std::fabs(dist[0 * n + 4] - 3.0f) < 1e-5f);

    std::cout << "  Calculated shortest distance 0 -> 5: " << dist[0 * n + 5] << " (expected 5.0 via shortcut)" << std::endl;
    assert(std::fabs(dist[0 * n + 5] - 5.0f) < 1e-5f);

    size_t candidates[] = {2, 4, 8};
    size_t best = weft::TropicalMetricRouter::select_best_peer(dist.data(), n, 0, candidates, 3);
    std::cout << "  Best peer among {2, 4, 8} from 0: " << best << " (expected 4)" << std::endl;
    assert(best == 4);

    std::cout << "[PASS] test_tropical_metric_networks succeeded." << std::endl;
    return 0;
}
