#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_gemm_lowbit_158.h"
#include "csrc/cli/weft_subbit_pack_engine.h"

int main() {
    std::cout << "[Test] Initializing GemmLowBit 1.58-bit Ternary Matrix-Vector SIMD Test..." << std::endl;

    const size_t M = 8;
    const size_t N = 32;

    std::vector<float> matrix_A(M * N, 0.0f);
    std::vector<float> x(N, 1.0f);
    std::vector<float> bias(M, 0.5f);
    std::vector<float> y(M, 0.0f);

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            if ((m + n) % 3 == 0) matrix_A[m * N + n] = 1.0f;
            else if ((m + n) % 3 == 1) matrix_A[m * N + n] = -1.0f;
            else matrix_A[m * N + n] = 0.0f;
        }
    }

    std::vector<uint8_t> packed_A(M * (N / 4), 0);
    for (size_t m = 0; m < M; ++m) {
        weft::avx512_pack_ternary_subbit(
            matrix_A.data() + m * N,
            packed_A.data() + m * (N / 4),
            N
        );
    }

    weft::avx512_gemv_158_bit(
        packed_A.data(),
        x.data(),
        bias.data(),
        y.data(),
        M,
        N
    );

    for (size_t m = 0; m < M; ++m) {
        float expected = bias[m];
        for (size_t n = 0; n < N; ++n) {
            expected += matrix_A[m * N + n] * x[n];
        }
        assert(std::fabs(y[m] - expected) < 1e-4f);
    }

    std::cout << "[Test] Output y[0]: " << y[0] << ", y[7]: " << y[7] << std::endl;
    std::cout << "[Test] GemmLowBit 1.58-bit Ternary Matrix-Vector SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
