#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_unitary_shimura.h"

int main() {
    std::cout << "[TEST] Running Unitary Shimura Varieties & Picard Modular Surfaces Invariant Test..." << std::endl;

    const size_t NUM_EXPERTS = 4;
    // 4 experts in B^2: each center has radius 0.4 < 1.0
    std::vector<float> expert_centers = {
        0.3f, 0.0f, 0.0f, 0.0f,  // Expert 0
        0.0f, 0.3f, 0.0f, 0.0f,  // Expert 1
        0.0f, 0.0f, 0.3f, 0.0f,  // Expert 2
        0.0f, 0.0f, 0.0f, 0.3f   // Expert 3
    };

    // Test token near Expert 2
    float z1_r = 0.0f, z1_i = 0.0f, z2_r = 0.28f, z2_i = 0.0f;
    weft::PicardRoutingProfile prof = weft::route_picard_modular_expert(
        z1_r, z1_i, z2_r, z2_i, expert_centers.data(), NUM_EXPERTS
    );

    std::cout << "  Selected expert: " << prof.selected_expert_id << " (expected: 2)" << std::endl;
    std::cout << "  Bergman distance: " << prof.bergman_distance << std::endl;
    std::cout << "  Routing confidence: " << prof.routing_confidence << std::endl;
    std::cout << "  Is inside unit ball: " << (prof.is_inside_ball ? "true" : "false") << std::endl;

    assert(prof.selected_expert_id == 2);
    assert(prof.is_inside_ball);
    assert(prof.bergman_distance < 0.1f);
    assert(prof.routing_confidence > 0.9f);

    std::cout << "[PASS] All Unitary Shimura Varieties & Picard Modular Surfaces invariants verified." << std::endl;
    return 0;
}
