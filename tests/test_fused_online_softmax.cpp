#include "csrc/cli/weft_fused_online_softmax.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    constexpr size_t size = 128;
    std::vector<float> logits(size);
    std::vector<float> probs(size, 0.0f);

    for (size_t i = 0; i < size; ++i) {
        logits[i] = static_cast<float>(i % 16) - 8.0f;
    }

    weft::fused_online_softmax_avx512(logits.data(), probs.data(), size);

    float sum_probs = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        assert(!std::isnan(probs[i]));
        assert(!std::isinf(probs[i]));
        assert(probs[i] >= 0.0f);
        sum_probs += probs[i];
    }

    assert(std::abs(sum_probs - 1.0f) < 0.01f);

    std::cout << "[PASS] Fused Online Softmax Unit Test passed (Sum probs: " 
              << sum_probs << ").\n";
    return 0;
}
