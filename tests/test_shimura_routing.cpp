#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_shimura_routing.h"

int main() {
    std::cout << "[TEST] Running Shimura Varieties Special Point Quantization Invariant Test..." << std::endl;

    const size_t DIM = 16;
    const size_t NUM_POINTS = 4;

    // CM Codebook with 4 orthogonal special points
    std::vector<float> codebook(NUM_POINTS * DIM, 0.0f);
    for (size_t p = 0; p < NUM_POINTS; ++p) {
        for (size_t d = p * 4; d < (p + 1) * 4; ++d) {
            codebook[p * DIM + d] = 1.0f;
        }
    }

    // Test 1: Weight vector identical to Point 2 -> closest is Point 2, distance 0
    std::vector<float> w2(DIM, 0.0f);
    for (size_t d = 8; d < 12; ++d) {
        w2[d] = 1.0f;
    }

    weft::ShimuraPointResult res2 = weft::route_to_shimura_special_point(w2.data(), codebook.data(), NUM_POINTS, DIM);
    std::cout << "  Closest CM index for w2: " << res2.closest_cm_index << " (distance: " << res2.killing_distance << ")" << std::endl;
    assert(res2.closest_cm_index == 2);
    assert(res2.killing_distance < 1e-5f);

    // Test 2: Weight vector closer to Point 0 than Point 1
    std::vector<float> w0(DIM, 0.0f);
    w0[0] = 0.9f;
    w0[1] = 0.8f;
    weft::ShimuraPointResult res0 = weft::route_to_shimura_special_point(w0.data(), codebook.data(), NUM_POINTS, DIM);
    std::cout << "  Closest CM index for w0: " << res0.closest_cm_index << " (distance: " << res0.killing_distance << ")" << std::endl;
    assert(res0.closest_cm_index == 0);
    assert(res0.killing_distance > 0.0f);

    std::cout << "[PASS] All Shimura Varieties Special Point Quantization invariants verified." << std::endl;
    return 0;
}
