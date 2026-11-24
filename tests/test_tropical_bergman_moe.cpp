#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_tropical_bergman_moe.h"

int main() {
    std::cout << "[Test] Initializing Tropical Bergman MoE Routing SIMD Test..." << std::endl;

    const size_t num_experts = 32;
    std::vector<float> valuations(num_experts, 0.0f);
    std::vector<float> out_distance(num_experts, 0.0f);

    float max_val = 0.0f;
    for (size_t i = 0; i < num_experts; ++i) {
        valuations[i] = static_cast<float>(i * 3);
        if (valuations[i] > max_val) {
            max_val = valuations[i];
        }
    }

    weft::avx512_tropical_bergman_routing(
        valuations.data(),
        out_distance.data(),
        num_experts,
        max_val
    );

    for (size_t i = 0; i < num_experts; ++i) {
        float expected = std::fabs(valuations[i] - max_val);
        assert(std::fabs(out_distance[i] - expected) < 1e-5f);
    }

    assert(out_distance[num_experts - 1] == 0.0f);
    std::cout << "[Test] Expert [31] distance to spine: " << out_distance[num_experts - 1] << std::endl;
    std::cout << "[Test] Tropical Bergman MoE Routing SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
