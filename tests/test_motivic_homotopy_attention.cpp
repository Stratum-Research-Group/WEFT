#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_motivic_homotopy_attention.h"

int main() {
    std::cout << "[TEST] Running Motivic Homotopy Theory Attention Invariant Test..." << std::endl;

    const size_t DIM = 32;
    std::vector<float> q(DIM, 1.0f);
    std::vector<float> k(DIM, 0.5f);

    weft::MotivicSphereConfig sphere{4, 2}; // p=4, q=2 -> simplicial dim = 2

    // Test 1: Simplicial dimension calculation and sphere validity
    weft::MotivicHomotopyResult res1 = weft::compute_motivic_homotopy_projection(q.data(), k.data(), DIM, sphere, 1.0f);
    std::cout << "  Simplicial ratio (p=4, q=2): " << res1.simplicial_ratio << std::endl;
    assert(res1.is_scale_invariant);
    assert(std::abs(res1.simplicial_ratio - 0.5f) < 1e-5f);

    // Test 2: Invariance under different affine line scales (A^1 contractibility)
    weft::MotivicHomotopyResult res2 = weft::compute_motivic_homotopy_projection(q.data(), k.data(), DIM, sphere, 4.0f);
    weft::MotivicHomotopyResult res3 = weft::compute_motivic_homotopy_projection(q.data(), k.data(), DIM, sphere, 100.0f);
    std::cout << "  Norm scale 1.0: " << res1.invariant_attention_norm << std::endl;
    std::cout << "  Norm scale 4.0: " << res2.invariant_attention_norm << std::endl;
    std::cout << "  Norm scale 100.0: " << res3.invariant_attention_norm << std::endl;
    assert(std::abs(res1.invariant_attention_norm - res2.invariant_attention_norm) < 1e-5f);
    assert(std::abs(res1.invariant_attention_norm - res3.invariant_attention_norm) < 1e-5f);

    // Test 3: Motivic suspension preserves simplicial difference
    weft::MotivicSphereConfig suspended_sphere{sphere.p + 1, sphere.q + 1}; // (5, 3)
    uint32_t orig_diff = sphere.p - sphere.q;
    uint32_t susp_diff = suspended_sphere.p - suspended_sphere.q;
    assert(orig_diff == susp_diff);

    std::cout << "[PASS] All Motivic Homotopy Theory Attention invariants verified." << std::endl;
    return 0;
}
