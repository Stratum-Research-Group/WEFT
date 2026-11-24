// test_speculative_dag_traversal.cpp
// Validation of AVX-512 speculative DAG pruning and distribution conservation.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_speculative_dag.h"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_speculative_dag_traversal..." << std::endl;

    const int count = 32;
    std::vector<float> probs(count);
    std::vector<int> tokens(count);

    // Populate candidate tokens with decaying likelihoods
    for (int i = 0; i < count; ++i) {
        probs[i] = 1.0f / (float(i) + 1.0f);
        tokens[i] = 1000 + i;
    }

    const float tau = 0.1f; // Retain only items with prob >= 0.1 (first 10 items)
    std::vector<int> surviving_ids(count);
    std::vector<float> surviving_probs(count);

    int surviving = weft::filter_speculative_dag_nodes_avx512(
        probs.data(), tokens.data(), count, tau,
        surviving_ids.data(), surviving_probs.data()
    );

    std::cout << "[TEST] Total Candidates: " << count << " | Retained Nodes: " << surviving << std::endl;
    assert(surviving == 10);

    for (int i = 0; i < surviving; ++i) {
        assert(surviving_probs[i] >= tau);
        assert(surviving_ids[i] == 1000 + i);
    }

    // Test exact acceptance evaluation
    bool accept1 = weft::evaluate_target_acceptance(0.8f, 0.5f, 0.9f); // ratio = 1.6 -> accept_prob = 1.0 -> accept
    assert(accept1 == true);

    bool accept2 = weft::evaluate_target_acceptance(0.4f, 0.8f, 0.6f); // ratio = 0.5 -> accept_prob = 0.5, rand 0.6 -> reject
    assert(accept2 == false);

    std::cout << "[TEST] All Speculative DAG Traversal tests passed successfully!" << std::endl;
    return 0;
}
