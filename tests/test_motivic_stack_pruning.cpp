#include <iostream>
#include <cassert>
#include <vector>
#include "csrc/cli/weft_motivic_stack_pruning.h"

int main() {
    std::cout << "[Test] Initializing Motivic Stack Pruning SIMD Test..." << std::endl;

    const size_t count = 32;
    const float threshold = 5.0f;

    std::vector<float> inertia_vals(count, 0.0f);
    std::vector<uint8_t> out_retained(count, 0);

    for (size_t i = 0; i < count; ++i) {
        inertia_vals[i] = static_cast<float>(i);
    }

    size_t retained = weft::avx512_stack_token_pruning(
        inertia_vals.data(),
        out_retained.data(),
        count,
        threshold
    );

    // Tokens 0..4 (< 5.0) must be 0
    // Tokens 5..31 (>= 5.0) must be 1 -> 27 retained
    assert(retained == 27);
    for (size_t i = 0; i < 5; ++i) assert(out_retained[i] == 0);
    for (size_t i = 5; i < count; ++i) assert(out_retained[i] == 1);

    std::cout << "[Test] Retained Tokens: " << retained << " / " << count << std::endl;
    std::cout << "[Test] Motivic Stack Pruning SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
