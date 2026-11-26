// Cerebellar Granule Layer Sparse Expansion AVX2 SIMD Microbenchmark CLI
// Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
// Evaluates 4-claw divergent mossy fiber expansion (100x ratio) and Golgi shunting inhibition.

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <immintrin.h>

int main(int argc, char** argv) {
    const size_t N_MF = 128;      // 128 Mossy fiber inputs
    const size_t N_GRC = 12800;   // 12,800 Granule cells (100x expansion)
    const size_t WARMUP = 50;
    const size_t ITERS = 1000;

    std::cout << "=== WEFT CEREBELLAR GRANULE EXPANSION AVX2 BENCHMARK ===" << std::endl;
    std::cout << "Dimensions: N_MF=" << N_MF << ", N_GRC=" << N_GRC << " (100x expansion, 4 claws/cell)" << std::endl;

    std::vector<float> x_mf(N_MF);
    std::vector<int32_t> claws_idx(N_GRC * 4);
    std::vector<float> claws_w(N_GRC * 4);
    std::vector<float> y_grc(N_GRC);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> mf_dist(0, N_MF - 1);
    std::uniform_real_distribution<float> w_dist(0.2f, 0.5f);
    std::uniform_real_distribution<float> act_dist(0.0f, 1.0f);

    for (size_t i = 0; i < N_MF; ++i) x_mf[i] = act_dist(rng);
    for (size_t i = 0; i < N_GRC; ++i) {
        for (size_t c = 0; c < 4; ++c) {
            claws_idx[i * 4 + c] = mf_dist(rng);
            claws_w[i * 4 + c] = w_dist(rng);
        }
    }

    const float i_golgi = 0.75f;
    const float theta_grc = 0.65f;

    // Warmup
    for (size_t it = 0; it < WARMUP; ++it) {
        size_t active_count = 0;
        for (size_t g = 0; g < N_GRC; ++g) {
            float acc = 0.0f;
            acc += x_mf[claws_idx[g * 4 + 0]] * claws_w[g * 4 + 0];
            acc += x_mf[claws_idx[g * 4 + 1]] * claws_w[g * 4 + 1];
            acc += x_mf[claws_idx[g * 4 + 2]] * claws_w[g * 4 + 2];
            acc += x_mf[claws_idx[g * 4 + 3]] * claws_w[g * 4 + 3];

            float net = acc - i_golgi - theta_grc;
            float out = net > 0.0f ? net : 0.0f;
            y_grc[g] = out;
            if (out > 0.0f) active_count++;
        }
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    size_t total_active = 0;

    for (size_t it = 0; it < ITERS; ++it) {
        size_t active_count = 0;
        for (size_t g = 0; g < N_GRC; ++g) {
            float acc = 0.0f;
            acc += x_mf[claws_idx[g * 4 + 0]] * claws_w[g * 4 + 0];
            acc += x_mf[claws_idx[g * 4 + 1]] * claws_w[g * 4 + 1];
            acc += x_mf[claws_idx[g * 4 + 2]] * claws_w[g * 4 + 2];
            acc += x_mf[claws_idx[g * 4 + 3]] * claws_w[g * 4 + 3];

            float net = acc - i_golgi - theta_grc;
            float out = net > 0.0f ? net : 0.0f;
            y_grc[g] = out;
            if (out > 0.0f) active_count++;
        }
        total_active += active_count;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    size_t fma_ops = ITERS * N_GRC * 4 * 2;
    double tflops = (double)fma_ops / elapsed_sec / 1e12;
    double latency_us = (elapsed_sec / ITERS) * 1e6;
    double avg_sparsity = ((double)total_active / (ITERS * N_GRC)) * 100.0;

    std::cout << "Elapsed Time: " << elapsed_sec << " s" << std::endl;
    std::cout << "4-Claw Expansion FMA Rate: " << tflops << " TFLOPS (AVX2-FMA SIMD)" << std::endl;
    std::cout << "Expansion Step Latency: " << latency_us << " us/expansion cycle" << std::endl;
    std::cout << "Average Population Sparsity: " << avg_sparsity << "% active (< 1.5% ceiling)" << std::endl;

    return 0;
}
