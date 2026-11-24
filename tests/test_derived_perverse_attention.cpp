#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_derived_perverse_attention.h"

int main() {
    std::cout << "[Test] Initializing Derived Perverse Attention Routing SIMD Test..." << std::endl;

    const size_t count = 32;
    const float delta = 0.5f;

    std::vector<float> raw_logits(count, 10.0f);
    std::vector<float> perverse_penalties(count, 0.0f);
    std::vector<float> out_logits(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        raw_logits[i] = 10.0f + static_cast<float>(i % 5);
        perverse_penalties[i] = static_cast<float>(i / 4);
    }

    weft::avx512_perverse_attention_routing(
        raw_logits.data(),
        perverse_penalties.data(),
        out_logits.data(),
        count,
        delta
    );

    for (size_t i = 0; i < count; ++i) {
        float expected = raw_logits[i] - delta * perverse_penalties[i];
        assert(std::fabs(out_logits[i] - expected) < 1e-5f);
    }

    std::cout << "[Test] Sample Logit [0]: " << out_logits[0] << std::endl;
    std::cout << "[Test] Sample Logit [31]: " << out_logits[31] << std::endl;
    std::cout << "[Test] Derived Perverse Attention Routing SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
