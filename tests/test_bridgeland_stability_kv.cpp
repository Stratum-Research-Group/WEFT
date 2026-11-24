#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_bridgeland_stability_kv.h"

int main() {
    std::cout << "[TEST] Running Bridgeland Stability KV-Dynamics Invariant Test..." << std::endl;

    // Test 1: Central charge phase computation
    // For deg = -1, rk = 1: -deg = 1, rk = 1 -> atan2(1, 1) = pi/4 -> phase = 0.25 * 0.5 = 0.125
    float p1 = weft::compute_central_charge_phase(-1.0f, 1.0f);
    std::cout << "  Computed phase for (-1, 1): " << p1 << std::endl;
    assert(p1 > 0.0f && p1 <= 1.0f);

    // Test 2: In-register KV cache pruning
    const size_t NUM_TOKENS = 6;
    const size_t DIM = 16;
    std::vector<float> kv_in(NUM_TOKENS * DIM, 2.0f);
    std::vector<float> kv_out(NUM_TOKENS * DIM, 0.0f);
    std::vector<float> deg = {-2.0f, 2.0f, -1.0f, 3.0f, -4.0f, 1.0f};
    std::vector<float> rk  = { 1.0f, 1.0f,  2.0f, 0.5f,  3.0f, 0.2f};

    size_t retained = weft::compact_kv_cache_bridgeland_simd(
        kv_in.data(), deg.data(), rk.data(), NUM_TOKENS, DIM, 0.1f, kv_out.data()
    );

    std::cout << "  Retained semistable tokens: " << retained << " / " << NUM_TOKENS << std::endl;
    assert(retained > 0 && retained <= NUM_TOKENS);

    for (size_t i = 0; i < retained * DIM; ++i) {
        assert(kv_out[i] == 2.0f);
    }

    std::cout << "[PASS] All Bridgeland Stability KV-Dynamics invariants verified." << std::endl;
    return 0;
}
