#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_condensed_liquid_kv.h"

int main() {
    std::cout << "[TEST] Running Condensed Mathematics & Liquid Tensor KV Invariant Test..." << std::endl;

    // Test 1: Liquid p-norm evaluation: 4 elements of value 4.0 -> sqrt(4) = 2. Sum = 8
    std::vector<float> vec = {4.0f, 4.0f, 4.0f, 4.0f};
    float l_norm = weft::compute_liquid_half_norm(vec.data(), vec.size());
    std::cout << "  Liquid 0.5-norm of [4, 4, 4, 4]: " << l_norm << " (Expected: 8.0)" << std::endl;
    assert(std::abs(l_norm - 8.0f) < 1e-5f);

    // Test 2: In-register KV cache pruning based on liquid norm
    const size_t NUM_TOKENS = 4;
    const size_t DIM = 16;
    std::vector<float> kv_in(NUM_TOKENS * DIM);
    std::vector<float> kv_out(NUM_TOKENS * DIM, 0.0f);

    for (size_t t = 0; t < NUM_TOKENS; ++t) {
        float val = (t % 2 == 0) ? 1.0f : 0.01f;
        for (size_t d = 0; d < DIM; ++d) {
            kv_in[t * DIM + d] = val;
        }
    }

    // tau_liquid = 8.0 -> retains tokens with val=1.0 (norm = 16 * 1 = 16), discards val=0.01 (norm = 16 * 0.1 = 1.6)
    size_t retained = weft::compact_kv_cache_liquid_simd(kv_in.data(), NUM_TOKENS, DIM, 8.0f, kv_out.data());
    std::cout << "  Retained liquid tokens (tau=8.0): " << retained << " / " << NUM_TOKENS << std::endl;
    assert(retained == 2);

    for (size_t i = 0; i < retained * DIM; ++i) {
        assert(kv_out[i] == 1.0f);
    }

    std::cout << "[PASS] All Condensed Liquid KV invariants verified." << std::endl;
    return 0;
}
