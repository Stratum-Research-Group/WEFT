#include "../csrc/cli/weft_cycle21_kernels.h"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "[TEST] Running Cycle 21 hardware kernels unit test..." << std::endl;

    // Test 1: Hodge-Tate twist scaling
    constexpr size_t N = 64;
    std::vector<float> in(N, 4.0f);
    std::vector<float> out(N, 0.0f);
    weft::apply_hodge_tate_twist_scaling(in.data(), out.data(), N, 2);
    for (size_t i = 0; i < N; ++i) {
        assert(std::fabs(out[i] - 1.0f) < 1e-5f);
    }
    std::cout << "  - apply_hodge_tate_twist_scaling: PASS" << std::endl;

    // Test 2: Tropical toric page table lookup
    std::vector<uint32_t> pt = {101, 202, 303, 404};
    uint32_t blk = weft::tropical_toric_page_lookup(pt.data(), pt.size(), 2);
    assert(blk == 303);
    std::cout << "  - tropical_toric_page_lookup: PASS" << std::endl;

    // Test 3: Monodromy speculative verification
    bool ok1 = weft::verify_speculative_monodromy(0.5f, 0.5f, 0.0f); // ratio 1.0 >= 0.85 -> true
    assert(ok1 == true);
    bool ok2 = weft::verify_speculative_monodromy(0.8f, 0.2f, 0.1f); // ratio 0.25 * exp(-0.1) < 0.85 -> false
    assert(ok2 == false);
    std::cout << "  - verify_speculative_monodromy: PASS" << std::endl;

    std::cout << "[DONE] All Cycle 21 kernel tests passed successfully." << std::endl;
    return 0;
}
