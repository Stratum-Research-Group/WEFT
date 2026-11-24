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

    // Adaptively build tree topology based on entropy H and temperature T
    WeftTreeTopology build_topology(float entropy, float temperature = 1.0f) {
        WeftTreeTopology topo;
        float effective_entropy = entropy * std::max(0.1f, temperature);

        if (effective_entropy < 0.5f) {
            // Low-entropy regime (Code, Math, Logic): Deep Linear Chain
            topo.depth = std::min(budget, static_cast<size_t>(8));
            topo.breadth = 1;
        } else if (effective_entropy < 1.5f) {
            // Medium-entropy regime: Balanced Tree
            topo.depth = 4;
            topo.breadth = budget / topo.depth;
        } else {
            // High-entropy regime (Creative, Dialogue): Broad Exploration Bush
            topo.depth = 2;
            topo.breadth = budget / topo.depth;
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
