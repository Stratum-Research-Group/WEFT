#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_nonabelian_hodge.h"

int main() {
    std::cout << "[TEST] Running Non-Abelian Hodge & Hitchin System Attention Invariant Test..." << std::endl;

    // Test 1: Invariant evaluation on simple diagonal matrix: {1, 2, 3}
    // Tr1 = 6, Tr1^2 = 36
    // Tr2 = 1 + 4 + 9 = 14
    // s2 = 0.5 * (36 - 14) = 11
    std::vector<float> diag = {1.0f, 2.0f, 3.0f};
    weft::HitchinSpectralState st = weft::compute_hitchin_invariants(diag.data(), 3);
    std::cout << "  Tr1=" << st.tr1 << ", Tr2=" << st.tr2 << ", s2=" << st.s2 << std::endl;
    assert(std::abs(st.s2 - 11.0f) < 1e-5f);

    // Test 2: Rescaling to target Hitchin invariant
    const size_t DIM = 32;
    std::vector<float> head_w(DIM, 1.0f);
    weft::rescale_head_to_hitchin_base_simd(head_w.data(), DIM, 44.0f);
    weft::HitchinSpectralState st_scaled = weft::compute_hitchin_invariants(head_w.data(), DIM);
    std::cout << "  Scaled s2: " << st_scaled.s2 << " (Target: 44.0)" << std::endl;
    assert(std::abs(st_scaled.s2 - 44.0f) < 1e-4f);

    std::cout << "[PASS] All Non-Abelian Hodge Attention invariants verified." << std::endl;
    return 0;
}
