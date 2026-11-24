#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_motivic_moduli_bundles.h"

int main() {
    std::cout << "[TEST] Running Motivic Moduli Attention Pruning Invariant Test..." << std::endl;

    // Test 1: Slope computation: deg=4, rk=2 -> mu = 2.0
    float mu = weft::compute_hn_slope(4.0f, 2.0f);
    std::cout << "  Computed HN slope: " << mu << " (Expected: 2.0)" << std::endl;
    assert(std::abs(mu - 2.0f) < 1e-6f);

    // Test 2: In-register head pruning: 4 heads
    const size_t NUM_HEADS = 4;
    const size_t HEAD_DIM = 16;
    std::vector<float> heads(NUM_HEADS * HEAD_DIM, 1.5f);
    std::vector<float> retained(NUM_HEADS * HEAD_DIM, 0.0f);

    std::vector<float> deg_max = {2.0f, 6.0f, 1.0f, 8.0f};
    std::vector<float> rk_max  = {1.0f, 2.0f, 1.0f, 2.0f}; // slopes: 2.0, 3.0, 1.0, 4.0
    std::vector<float> deg_min = {1.0f, 1.0f, 0.8f, 1.0f};
    std::vector<float> rk_min  = {1.0f, 2.0f, 1.0f, 2.0f}; // slopes: 1.0, 0.5, 0.8, 0.5
    // diffs: 1.0, 2.5, 0.2, 3.5

    // tau_hn = 1.5 -> retains heads 0 (diff=1.0) and 2 (diff=0.2) -> 2 heads
    size_t count = weft::prune_unstable_attention_heads_simd(
        heads.data(), deg_max.data(), rk_max.data(), deg_min.data(), rk_min.data(),
        NUM_HEADS, HEAD_DIM, 1.5f, retained.data()
    );

    std::cout << "  Retained semistable heads: " << count << " / " << NUM_HEADS << std::endl;
    assert(count == 2);

    for (size_t i = 0; i < count * HEAD_DIM; ++i) {
        assert(retained[i] == 1.5f);
    }

    std::cout << "[PASS] All Motivic Moduli Attention Pruning invariants verified." << std::endl;
    return 0;
}
