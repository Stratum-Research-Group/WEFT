// Hippocampal CA1 Dual-Stream SIMD AVX2 Microbenchmark CLI
// Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
// Evaluates dual-stream Schaffer (CA3) and Temporoammonic (EC) synaptic integration and phase gating.

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <immintrin.h>

int main(int argc, char** argv) {
    const size_t N_CA1 = 1024;
    const size_t N_CA3 = 512;
    const size_t N_EC  = 512;
    const size_t WARMUP = 100;
    const size_t ITERS  = 2000;

    std::cout << "=== WEFT CA1 DUAL-STREAM COMPARATOR AVX2 BENCHMARK ===" << std::endl;
    std::cout << "Network Dimension: N_CA1=" << N_CA1 << ", N_CA3=" << N_CA3 << ", N_EC=" << N_EC << std::endl;

    // Allocate memory
    std::vector<float> W_schaffer(N_CA1 * N_CA3);
    std::vector<float> W_ta(N_CA1 * N_EC);
    std::vector<float> x_ca3(N_CA3);
    std::vector<float> x_ec(N_EC);
    std::vector<float> y_ca1(N_CA1);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(N_CA3));
    std::uniform_real_distribution<float> act_dist(0.0f, 1.0f);

    for (size_t i = 0; i < N_CA1 * N_CA3; ++i) W_schaffer[i] = dist(rng);
    for (size_t i = 0; i < N_CA1 * N_EC; ++i)  W_ta[i] = dist(rng);
    for (size_t i = 0; i < N_CA3; ++i) x_ca3[i] = act_dist(rng);
    for (size_t i = 0; i < N_EC; ++i)  x_ec[i] = act_dist(rng);

    float theta_rad = 3.14159265f; // Trough phase (retrieval)
    float g_ca3 = 0.5f * (1.0f - std::cos(theta_rad));
    float g_ec  = 0.5f * (1.0f + std::cos(theta_rad));
    float inhibition = 0.05f;

    // Warmup
    for (size_t it = 0; it < WARMUP; ++it) {
        for (size_t i = 0; i < N_CA1; ++i) {
            __m256 acc_ca3 = _mm256_setzero_ps();
            for (size_t j = 0; j < N_CA3; j += 8) {
                __m256 w = _mm256_loadu_ps(&W_schaffer[i * N_CA3 + j]);
                __m256 x = _mm256_loadu_ps(&x_ca3[j]);
                acc_ca3 = _mm256_fmadd_ps(w, x, acc_ca3);
            }
            alignas(32) float tmp3[8];
            _mm256_storeu_ps(tmp3, acc_ca3);
            float s3 = tmp3[0] + tmp3[1] + tmp3[2] + tmp3[3] + tmp3[4] + tmp3[5] + tmp3[6] + tmp3[7];

            __m256 acc_ec = _mm256_setzero_ps();
            for (size_t j = 0; j < N_EC; j += 8) {
                __m256 w = _mm256_loadu_ps(&W_ta[i * N_EC + j]);
                __m256 x = _mm256_loadu_ps(&x_ec[j]);
                acc_ec = _mm256_fmadd_ps(w, x, acc_ec);
            }
            alignas(32) float tmp_ec[8];
            _mm256_storeu_ps(tmp_ec, acc_ec);
            float s_ec = tmp_ec[0] + tmp_ec[1] + tmp_ec[2] + tmp_ec[3] + tmp_ec[4] + tmp_ec[5] + tmp_ec[6] + tmp_ec[7];

            float net = g_ca3 * s3 + g_ec * s_ec - inhibition;
            y_ca1[i] = net > 0.0f ? net : 0.0f;
        }
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < ITERS; ++it) {
        for (size_t i = 0; i < N_CA1; ++i) {
            __m256 acc_ca3 = _mm256_setzero_ps();
            for (size_t j = 0; j < N_CA3; j += 8) {
                __m256 w = _mm256_loadu_ps(&W_schaffer[i * N_CA3 + j]);
                __m256 x = _mm256_loadu_ps(&x_ca3[j]);
                acc_ca3 = _mm256_fmadd_ps(w, x, acc_ca3);
            }
            alignas(32) float tmp3[8];
            _mm256_storeu_ps(tmp3, acc_ca3);
            float s3 = tmp3[0] + tmp3[1] + tmp3[2] + tmp3[3] + tmp3[4] + tmp3[5] + tmp3[6] + tmp3[7];

            __m256 acc_ec = _mm256_setzero_ps();
            for (size_t j = 0; j < N_EC; j += 8) {
                __m256 w = _mm256_loadu_ps(&W_ta[i * N_EC + j]);
                __m256 x = _mm256_loadu_ps(&x_ec[j]);
                acc_ec = _mm256_fmadd_ps(w, x, acc_ec);
            }
            alignas(32) float tmp_ec[8];
            _mm256_storeu_ps(tmp_ec, acc_ec);
            float s_ec = tmp_ec[0] + tmp_ec[1] + tmp_ec[2] + tmp_ec[3] + tmp_ec[4] + tmp_ec[5] + tmp_ec[6] + tmp_ec[7];

            float net = g_ca3 * s3 + g_ec * s_ec - inhibition;
            y_ca1[i] = net > 0.0f ? net : 0.0f;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    size_t total_synaptic_ops = ITERS * N_CA1 * (N_CA3 + N_EC) * 2;
    double tflops = (double)total_synaptic_ops / elapsed_sec / 1e12;
    double latency_us = (elapsed_sec / ITERS) * 1e6;

    std::cout << "Elapsed Time: " << elapsed_sec << " s" << std::endl;
    std::cout << "Dual-Stream Synaptic FMA Rate: " << tflops << " TFLOPS (AVX2-FMA SIMD)" << std::endl;
    std::cout << "CA1 Comparator Latency: " << latency_us << " us/readout cycle" << std::endl;

    return 0;
}
