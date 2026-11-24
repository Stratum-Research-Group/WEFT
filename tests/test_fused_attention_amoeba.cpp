#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_fused_attention_amoeba.h"

int main() {
    std::cout << "[Test] Initializing Fused Attention Amoeba SIMD Test..." << std::endl;

    const size_t count = 32;
    const float ceiling = 5.0f;

    std::vector<float> raw_logits(count, 0.0f);
    std::vector<float> bounded_logits(count, 0.0f);
    float out_max = 0.0f;
    float out_sum = 0.0f;

    for (size_t i = 0; i < count; ++i) {
        raw_logits[i] = static_cast<float>(i % 10);
    }

    weft::avx512_fused_attention_amoeba_tile(
        raw_logits.data(),
        bounded_logits.data(),
        &out_max,
        &out_sum,
        count,
        ceiling
    );

    assert(out_max <= ceiling);
    assert(out_sum > 0.0f);
    for (size_t i = 0; i < count; ++i) {
        assert(bounded_logits[i] <= ceiling);
    }

    std::cout << "[Test] Max Bounded Logit: " << out_max << ", Partition Sum: " << out_sum << std::endl;
    std::cout << "[Test] Fused Attention Amoeba SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
