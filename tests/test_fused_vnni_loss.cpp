// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_fused_vnni_loss.h"

int main() {
    std::cout << "[TEST] Running Fused VNNI Loss Engine AVX-512 Test..." << std::endl;

    const size_t vocab_size = 128;
    const size_t target_idx = 42;
    std::vector<float> logits(vocab_size);

    for (size_t i = 0; i < vocab_size; ++i) {
        logits[i] = 0.05f * (i % 10);
    }
    logits[target_idx] = 5.0f; // High confidence target

    float loss = weft::FusedVnniLossEngine::compute_cross_entropy(logits.data(), vocab_size, target_idx);
    assert(loss >= 0.0f);
    assert(loss < 1.0f); // loss is ~0.732 for target logit 5.0 vs 127 other logits up to 0.45

    std::cout << "[PASS] Fused VNNI Loss Engine verified successfully (loss: " << loss << ")." << std::endl;
    return 0;
}
