#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "csrc/cli/weft_tropical_toric_fiber_products.h"

int main() {
    std::cout << "[TEST] Running test_tropical_toric_fiber_products (Paper 313)..." << std::endl;

    const size_t pages = 16;
    std::vector<float> latencies(pages, 1.0f);
    std::vector<float> freqs(pages, 10.0f);
    std::vector<float> scores(pages, 0.0f);
    float weights[2] = {1.0f, 2.0f};

    // Page 7 has very low access frequency and high latency -> lowest utility
    freqs[7] = 0.01f;
    latencies[7] = 50.0f;

    size_t victim = weft::TropicalFiberPager::select_eviction_candidate(latencies.data(), freqs.data(), weights, pages);
    std::cout << "  Selected eviction victim: " << victim << " (expected 7)" << std::endl;
    assert(victim == 7);

    // Test evaluate_eviction_scores where latency weight is 0.0f to test pure frequency scoring
    weft::TropicalFiberPager::evaluate_eviction_scores(freqs.data(), latencies.data(), scores.data(), pages, 1.0f, 0.0f);
    assert(scores[0] > scores[7]);
    std::cout << "  Score of page 0: " << scores[0] << ", score of victim 7: " << scores[7] << std::endl;

    std::cout << "[PASS] test_tropical_toric_fiber_products succeeded." << std::endl;
    return 0;
}
