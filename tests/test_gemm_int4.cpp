#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_gemm_int4.h"

int main() {
    std::cout << "[Test] Initializing GemmInt4 SIMD Test..." << std::endl;

    const size_t M = 4;
    const size_t N = 16;

    std::vector<uint8_t> packed_A(M * (N / 2), 0);
    std::vector<float> x(N, 1.0f);
    std::vector<float> scales(M, 0.5f);
    std::vector<float> bias(M, 1.0f);
    std::vector<float> y(M, 0.0f);

    // Pack weights: low nibble = 10 (val = +2), high nibble = 6 (val = -2)
    // byte = 0x6A
    for (size_t i = 0; i < packed_A.size(); ++i) {
        packed_A[i] = 0x6A;
    }

    weft::avx512_gemv_int4(
        packed_A.data(),
        x.data(),
        scales.data(),
        bias.data(),
        y.data(),
        M,
        N
    );

    // Each byte has (+2) + (-2) = 0 dot contribution
    // y[m] = scale * 0 + bias[m] = 1.0f
    for (size_t m = 0; m < M; ++m) {
        assert(std::fabs(y[m] - 1.0f) < 1e-5f);
    }

    std::cout << "[Test] Output y[0]: " << y[0] << ", y[3]: " << y[3] << std::endl;
    std::cout << "[Test] GemmInt4 SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
