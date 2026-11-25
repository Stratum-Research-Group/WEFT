#include "csrc/cli/weft_derived_tannakian_categories.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    constexpr size_t num_patches = 4;
    constexpr size_t dim = 64;

    std::vector<float> visual_patches(num_patches * dim, 1.0f);
    std::vector<int8_t> text_ternary(dim, 1);
    std::vector<float> fused_out(num_patches, 0.0f);

    weft::tannakian_multimodal_fusion_avx512(
        visual_patches.data(), text_ternary.data(), fused_out.data(),
        num_patches, dim, 0.5f);

    // Expected: 64 * 1.0 * 1 * 0.5 = 32.0f per patch
    for (size_t p = 0; p < num_patches; ++p) {
        assert(!std::isnan(fused_out[p]));
        assert(std::abs(fused_out[p] - 32.0f) < 1e-4f);
    }

    std::cout << "[PASS] Derived Tannakian Multi-Modal Fusion Unit Test passed (Fused score: " 
              << fused_out[0] << ").\n";
    return 0;
}
