#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_perverse_sheaves.h"

int main() {
    std::cout << "[TEST] Running Perverse Sheaves & BBDG Decomposition Invariant Test..." << std::endl;

    // Test 1: Middle perversity support condition check
    // Dim 2, Deg -2 -> sum = 0 <= 0 (True)
    // Dim 2, Deg -1 -> sum = 1 > 0 (False)
    bool s1 = weft::check_middle_perversity(2, -2);
    bool s2 = weft::check_middle_perversity(2, -1);
    std::cout << "  Support check: (2, -2)=" << s1 << ", (2, -1)=" << s2 << std::endl;
    assert(s1 == true);
    assert(s2 == false);

    // Test 2: In-register gradient projection preserving perverse support
    const size_t DIM = 4;
    std::vector<float> raw_grads = {1.5f, 2.0f, 3.5f, 4.0f};
    std::vector<int32_t> dims = {1, 2, 3, 4};
    std::vector<int32_t> degs = {-2, -1, -3, -5}; // sums: -1 (<=0), 1 (>0), 0 (<=0), -1 (<=0)
    std::vector<float> out_grads(DIM, 0.0f);

    weft::project_perverse_stratum_gradients_simd(
        raw_grads.data(), dims.data(), degs.data(), DIM, out_grads.data()
    );

    assert(out_grads[0] == 1.5f);
    assert(out_grads[1] == 0.0f); // Suppressed off-support
    assert(out_grads[2] == 3.5f);
    assert(out_grads[3] == 4.0f);

    std::cout << "  Perverse support filtering correctly verified." << std::endl;
    std::cout << "[PASS] All Perverse Sheaves invariants verified." << std::endl;
    return 0;
}
