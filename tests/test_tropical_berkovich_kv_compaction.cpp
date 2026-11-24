#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_tropical_berkovich_kv_compaction.h"

int main() {
    std::cout << "[Test] Initializing Tropical Berkovich KV Compaction SIMD Test..." << std::endl;

    const size_t num_pages = 32;
    const float threshold = 0.5f;

    std::vector<float> occupancy(num_pages, 0.0f);
    std::vector<float> out_urgency(num_pages, 0.0f);

    for (size_t i = 0; i < num_pages; ++i) {
        occupancy[i] = static_cast<float>(i) / static_cast<float>(num_pages);
    }

    weft::avx512_berkovich_compaction_urgency(
        occupancy.data(),
        out_urgency.data(),
        num_pages,
        threshold
    );

    for (size_t i = 0; i < num_pages; ++i) {
        float expected = threshold - occupancy[i];
        if (expected < 0.0f) expected = 0.0f;
        assert(std::fabs(out_urgency[i] - expected) < 1e-5f);
    }

    std::cout << "[Test] Urgency [0]: " << out_urgency[0] << std::endl;
    std::cout << "[Test] Urgency [31]: " << out_urgency[31] << " (0 as occupancy > threshold)" << std::endl;
    std::cout << "[Test] Tropical Berkovich KV Compaction SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
