#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_tropical_wronskian_pruning.h"

int main() {
    std::cout << "[Test] Initializing Tropical Wronskian Head Pruning SIMD Test..." << std::endl;

    const size_t count = 64;
    std::vector<float> head_a(count, 0.0f);
    std::vector<float> head_b(count, 0.0f);
    std::vector<float> out_diff(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        head_a[i] = static_cast<float>(i * 2);
        head_b[i] = static_cast<float>(i * 2 + 1);
    }
    // inject a larger difference at index 10
    head_a[10] = 50.0f;
    head_b[10] = 10.0f;

    float max_valuation = weft::avx512_tropical_wronskian_valuation(
        head_a.data(),
        head_b.data(),
        out_diff.data(),
        count
    );

    assert(std::fabs(max_valuation - 40.0f) < 1e-5f);
    assert(std::fabs(out_diff[10] - 40.0f) < 1e-5f);
    assert(std::fabs(out_diff[0] - 1.0f) < 1e-5f);

    std::cout << "[Test] Max Tropical Wronskian Valuation: " << max_valuation << std::endl;
    std::cout << "[Test] Tropical Wronskian Head Pruning SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
