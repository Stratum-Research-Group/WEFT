#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_tropical_jacobian_attention.h"

int main() {
    std::cout << "[Test] Initializing Tropical Jacobian Attention SIMD Test..." << std::endl;

    const size_t count = 32;
    std::vector<float> vision_coords(count, 0.0f);
    std::vector<float> text_coords(count, 0.0f);
    std::vector<float> out_dists(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        vision_coords[i] = static_cast<float>(i * 2);
        text_coords[i] = static_cast<float>(i + 5);
    }

    weft::avx512_tropical_jacobian_distance(
        vision_coords.data(),
        text_coords.data(),
        out_dists.data(),
        count
    );

    for (size_t i = 0; i < count; ++i) {
        float expected = std::fabs(vision_coords[i] - text_coords[i]);
        assert(std::fabs(out_dists[i] - expected) < 1e-5f);
    }

    std::cout << "[Test] Dist [0]: " << out_dists[0] << ", Dist [31]: " << out_dists[31] << std::endl;
    std::cout << "[Test] Tropical Jacobian Attention SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
