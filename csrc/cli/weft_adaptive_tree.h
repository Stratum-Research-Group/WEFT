#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>

namespace weft {

struct WeftTreeTopology {
    size_t depth;
    size_t breadth;
    size_t total_nodes;
    float effective_temperature;
    std::vector<int32_t> parent_indices; // Parent index for each tree node
};

class WeftAdaptiveTreeBuilder {
public:
    size_t budget;

    WeftAdaptiveTreeBuilder(size_t budget = 16) : budget(budget) {}

    // Compute Shannon entropy from draft probability distribution
    static float compute_entropy(const float* probs, size_t vocab_size) {
        float h = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            if (probs[i] > 1e-7f) {
                h -= probs[i] * std::log2(probs[i]);
            }
        }
        return h;
    }

    // Adaptively build tree topology based on entropy H and temperature T with continuous non-linear scaling
    WeftTreeTopology build_topology(float entropy, float temperature = 1.0f) {
        WeftTreeTopology topo;
        float safe_temp = std::max(0.01f, std::min(temperature, 2.5f));
        topo.effective_temperature = safe_temp;

        // Dynamic scaling factor: Higher temp magnifies entropy spread; Low temp freezes into single greedy beam
        float effective_entropy = entropy * safe_temp;

        if (safe_temp <= 0.15f || effective_entropy < 0.45f) {
            // Greedy / Deterministic regime (Formal math, verified code): Deep Linear Chain (Breadth = 1)
            topo.depth = std::min(budget, static_cast<size_t>(8));
            topo.breadth = 1;
        } else if (effective_entropy < 1.4f) {
            // Balanced cognitive flow: 4-deep multi-branch tree
            topo.depth = 4;
            topo.breadth = std::max(static_cast<size_t>(1), budget / topo.depth);
        } else {
            // High-entropy exploration (Brainstorming, speculative creative paths): Wide 2-deep bush
            topo.depth = 2;
            topo.breadth = std::max(static_cast<size_t>(1), budget / topo.depth);
        }

        topo.total_nodes = std::min(budget, topo.depth * topo.breadth);
        topo.parent_indices.resize(topo.total_nodes);

        // Populate causal parent indices
        for (size_t i = 0; i < topo.total_nodes; ++i) {
            if (topo.breadth == 1) {
                topo.parent_indices[i] = (i == 0) ? -1 : static_cast<int32_t>(i - 1);
            } else {
                size_t d = i / topo.breadth;
                topo.parent_indices[i] = (d == 0) ? -1 : static_cast<int32_t>((d - 1) * topo.breadth);
            }
        }

        return topo;
    }
};

} // namespace weft
