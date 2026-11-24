#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_tropical_convexity_kv.h"

int main() {
    std::cout << "[Test] Initializing Tropical Convexity KV Isolation SIMD Test..." << std::endl;

    const size_t count = 32;
    std::vector<float> query_addrs(count, 0.0f);
    std::vector<float> page_base(count, 100.0f);
    std::vector<float> page_size(count, 50.0f);
    std::vector<float> out_violations(count, 0.0f);

    // Queries inside [100, 150) -> violation 0
    // Queries outside -> violation > 0
    for (size_t i = 0; i < count; ++i) {
        query_addrs[i] = 80.0f + static_cast<float>(i * 3);
    }

    weft::avx512_tropical_polytope_isolation(
        query_addrs.data(),
        page_base.data(),
        page_size.data(),
        out_violations.data(),
        count
    );

    for (size_t i = 0; i < count; ++i) {
        float above = std::max(0.0f, query_addrs[i] - 150.0f);
        float below = std::max(0.0f, 100.0f - query_addrs[i]);
        assert(std::fabs(out_violations[i] - (above + below)) < 1e-5f);
    }

    std::cout << "[Test] Query [0] violation: " << out_violations[0] << " (addr 80 < 100)" << std::endl;
    std::cout << "[Test] Query [10] violation: " << out_violations[10] << " (addr 110 in [100, 150))" << std::endl;
    std::cout << "[Test] Tropical Convexity KV Isolation SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
