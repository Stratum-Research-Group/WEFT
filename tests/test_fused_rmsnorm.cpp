#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_fused_rmsnorm.h"

int main() {
    std::cout << "[Test] Initializing Fused RMSNorm SIMD Test..." << std::endl;

    const size_t dim = 32;
    std::vector<float> x(dim, 0.0f);
    std::vector<float> weight(dim, 1.0f);
    std::vector<float> y(dim, 0.0f);

    for (size_t i = 0; i < dim; ++i) {
        x[i] = static_cast<float>(i + 1);
        weight[i] = 1.0f;
    }

    weft::avx512_fused_rmsnorm(
        x.data(),
        weight.data(),
        y.data(),
        dim,
        1e-6f
    );

    // Compute expected manual RMS
    float sq_sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) sq_sum += x[i] * x[i];
    float rms = std::sqrt(sq_sum / static_cast<float>(dim) + 1e-6f);

    for (size_t i = 0; i < dim; ++i) {
        float expected = (x[i] / rms) * weight[i];
        assert(std::fabs(y[i] - expected) < 1e-4f);
    }

    std::cout << "[Test] RMSNorm y[0]: " << y[0] << ", y[31]: " << y[31] << std::endl;
    std::cout << "[Test] Fused RMSNorm SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
