#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include "../csrc/cli/weft_avx512_multihead_verify.h"

// Standalone C++ Microbenchmark for Multi-Head Parallel Verification
// Measures speedup of weft_avx512_multihead_verify against standard scalar loops.

void scalar_multihead_verify(const float* q_heads, const float* k_heads, float* out_scores) {
    const float scale = 1.0f / std::sqrt(64.0f);
    for (size_t h = 0; h < 8; ++h) {
        float dot = 0.0f;
        for (size_t d = 0; d < 64; ++d) {
            dot += q_heads[h * 64 + d] * k_heads[h * 64 + d];
        }
        out_scores[h] = dot * scale;
    }
}

int main() {
    std::cout << "=== WEFT Multi-Head Bitboard Parallel Verification Microbenchmark ===" << std::endl;
    constexpr size_t NUM_HEADS = 8;
    constexpr size_t HEAD_DIM = 64;
    constexpr size_t TOTAL_FLOATS = NUM_HEADS * HEAD_DIM;
    constexpr size_t ITERS = 50000;

    std::vector<float> q(TOTAL_FLOATS, 0.25f);
    std::vector<float> k(TOTAL_FLOATS, 0.50f);
    std::vector<float> out_simd(NUM_HEADS, 0.0f);
    std::vector<float> out_scalar(NUM_HEADS, 0.0f);

    // Warmup
    for (size_t i = 0; i < 500; ++i) {
        weft::WeftAVX512MultiHeadVerifier::verify_8heads_parallel(q.data(), k.data(), out_simd.data());
        scalar_multihead_verify(q.data(), k.data(), out_scalar.data());
    }

    // Benchmark SIMD
    auto start_simd = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERS; ++i) {
        weft::WeftAVX512MultiHeadVerifier::verify_8heads_parallel(q.data(), k.data(), out_simd.data());
    }
    auto end_simd = std::chrono::high_resolution_clock::now();
    double time_simd_ms = std::chrono::duration<double, std::milli>(end_simd - start_simd).count();

    // Benchmark Scalar
    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERS; ++i) {
        scalar_multihead_verify(q.data(), k.data(), out_scalar.data());
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    double time_scalar_ms = std::chrono::duration<double, std::milli>(end_scalar - start_scalar).count();

    double ns_per_simd = (time_simd_ms * 1e6) / ITERS;
    double ns_per_scalar = (time_scalar_ms * 1e6) / ITERS;
    double speedup = time_scalar_ms / time_simd_ms;

    std::cout << "[RESULT] Total Iterations : " << ITERS << std::endl;
    std::cout << "[RESULT] SIMD Latency    : " << ns_per_simd << " ns per 8-head step" << std::endl;
    std::cout << "[RESULT] Scalar Latency  : " << ns_per_scalar << " ns per 8-head step" << std::endl;
    std::cout << "[RESULT] Speedup Ratio   : " << speedup << "x faster" << std::endl;
    std::cout << "[RESULT] Verification Checksum (Head 0): SIMD=" << out_simd[0] << ", Scalar=" << out_scalar[0] << std::endl;

    return 0;
}
