#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_tropical_grassmannian_pruning.h"

int main() {
    std::cout << "[Test] Initializing Tropical Grassmannian Pruning SIMD Test..." << std::endl;

    const size_t count = 32;

    std::vector<float> pik(count, 0.0f);
    std::vector<float> pjl(count, 0.0f);
    std::vector<float> pil(count, 0.0f);
    std::vector<float> pjk(count, 0.0f);
    std::vector<float> out_diff(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        pik[i] = static_cast<float>(i + 1);
        pjl[i] = static_cast<float>(i + 2);
        pil[i] = static_cast<float>(i);
        pjk[i] = static_cast<float>(i + 1);
    }
    // (pik + pjl) = 2i + 3, (pil + pjk) = 2i + 1 -> diff = 2.0

    weft::avx512_tropical_plucker_discrepancy(
        pik.data(),
        pjl.data(),
        pil.data(),
        pjk.data(),
        out_diff.data(),
        count
    );

    for (size_t i = 0; i < count; ++i) {
        assert(std::fabs(out_diff[i] - 2.0f) < 1e-5f);
    }

    std::cout << "[Test] Discrepancy [0]: " << out_diff[0] << ", Discrepancy [31]: " << out_diff[31] << std::endl;
    std::cout << "[Test] Tropical Grassmannian Pruning SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
