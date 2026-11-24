#include <iostream>
#include <cassert>
#include <vector>
#include <numeric>
#include "../csrc/cli/weft_pregate_moe.h"

int main() {
    std::cout << "[TEST] Running Weft Pre-Gated MoE Routing Verification..." << std::endl;

    weft::WeftPreGatedMoERouter router(160, 8);

    std::vector<float> logits(160, 0.0f);
    // Put highest scores at indices 10, 20, 30, 40, 50, 60, 70, 80
    logits[10] = 5.0f;
    logits[20] = 4.5f;
    logits[30] = 4.0f;
    logits[40] = 3.5f;
    logits[50] = 3.0f;
    logits[60] = 2.5f;
    logits[70] = 2.0f;
    logits[80] = 1.5f;

    auto decision = router.compute_topk_routing(logits.data());
    assert(decision.expert_indices[0] == 10);
    assert(decision.expert_indices[1] == 20);
    assert(decision.expert_indices[7] == 80);

    float sum_w = 0.0f;
    for (size_t i = 0; i < 8; ++i) {
        sum_w += decision.expert_weights[i];
    }
    assert(std::abs(sum_w - 1.0f) < 1e-5f);
    std::cout << "[PASS] Top-8 routing indices and normalized weights verified." << std::endl;

    // Test fused expert outputs
    const size_t dim = 64;
    std::vector<std::vector<float>> expert_outs(8, std::vector<float>(dim, 1.0f));
    std::vector<const float*> expert_ptrs(8);
    for (size_t e = 0; e < 8; ++e) expert_ptrs[e] = expert_outs[e].data();

    std::vector<float> final_out(dim, 0.0f);
    router.fuse_expert_outputs(expert_ptrs.data(), decision.expert_weights.data(), dim, final_out.data());

    // Since each expert outputs 1.0f, and sum(weights) == 1.0f, final_out should be 1.0f
    for (size_t i = 0; i < dim; ++i) {
        assert(std::abs(final_out[i] - 1.0f) < 1e-4f);
    }
    std::cout << "[PASS] Fused expert outputs in AVX-512 verified." << std::endl;

    std::cout << "[ALL PASSED] Pre-Gated MoE tests successful." << std::endl;
    return 0;
}
