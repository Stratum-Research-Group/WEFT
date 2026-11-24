#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_gemm_lowbit_batch.h"
#include "csrc/cli/weft_subbit_pack_engine.h"

int main() {
    std::cout << "[Test] Initializing GemmLowBit Batch Dispatcher SIMD Test..." << std::endl;

    const size_t batch_size = 8;
    const size_t M = 16;
    const size_t N = 32;

    std::vector<float> matrix_A(M * N, 0.0f);
    std::vector<float> X_batch(batch_size * N, 1.0f);
    std::vector<float> bias(M, 0.2f);
    std::vector<float> Y_batch(batch_size * M, 0.0f);

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            if ((m + n) % 3 == 0) matrix_A[m * N + n] = 1.0f;
            else if ((m + n) % 3 == 1) matrix_A[m * N + n] = -1.0f;
            else matrix_A[m * N + n] = 0.0f;
        }
    }

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t n = 0; n < N; ++n) {
            X_batch[b * N + n] = static_cast<float>(b + 1) * 0.1f;
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

    weft::avx512_gemm_lowbit_batch_dispatch(
        packed_A.data(),
        X_batch.data(),
        bias.data(),
        Y_batch.data(),
        batch_size,
        M,
        N,
        4 // 4 worker threads
    );

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t m = 0; m < M; ++m) {
            float expected = bias[m];
            for (size_t n = 0; n < N; ++n) {
                expected += matrix_A[m * N + n] * X_batch[b * N + n];
            }
            assert(std::fabs(Y_batch[b * M + m] - expected) < 1e-4f);
        }
    }

    std::cout << "[Test] Batch 0 Y[0]: " << Y_batch[0] << ", Batch 7 Y[0]: " << Y_batch[7 * M] << std::endl;
    std::cout << "[Test] GemmLowBit Batch Dispatcher SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
