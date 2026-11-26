// Subicular Burst Broadcasting SIMD AVX2 Microbenchmark CLI
// Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
// Evaluates burst generation and parallel broadcasting to mPFC, deep EC, and mammillary bodies.

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <immintrin.h>

int main(int argc, char** argv) {
    const size_t N_SUB = 1024;    // 1024 Subicular pyramidal cells
    const size_t N_MPFC = 512;    // 512 mPFC target neurons
    const size_t N_EC_DEEP = 512; // 512 deep EC target neurons
    const size_t WARMUP = 100;
    const size_t ITERS = 2000;

    std::cout << "=== WEFT SUBICULUM BROADCASTING AVX2 BENCHMARK ===" << std::endl;
    std::cout << "Network Dimension: N_SUB=" << N_SUB << ", Targets: mPFC=" << N_MPFC << ", EC_DEEP=" << N_EC_DEEP << std::endl;

    std::vector<float> v_in(N_SUB);
    std::vector<float> burst_act(N_SUB);
    std::vector<float> W_mpfc(N_MPFC * N_SUB);
    std::vector<float> W_ec(N_EC_DEEP * N_SUB);
    std::vector<float> y_mpfc(N_MPFC);
    std::vector<float> y_ec(N_EC_DEEP);

    std::mt19937 rng(42);
    std::normal_distribution<float> w_dist(0.0f, 1.0f / std::sqrt(N_SUB));
    std::uniform_real_distribution<float> v_dist(-65.0f, -45.0f); // Range around burst threshold -55 mV

    for (size_t i = 0; i < N_SUB; ++i) v_in[i] = v_dist(rng);
    for (size_t i = 0; i < N_MPFC * N_SUB; ++i) W_mpfc[i] = w_dist(rng);
    for (size_t i = 0; i < N_EC_DEEP * N_SUB; ++i) W_ec[i] = w_dist(rng);

    const float v_thresh = -55.0f;
    const float g_T = 1.5f;
    const float E_Ca = 120.0f;
    const float swr_amp = 1.85f;

    // Warmup
    for (size_t it = 0; it < WARMUP; ++it) {
        // 1. Compute burst activation
        for (size_t i = 0; i < N_SUB; ++i) {
            float v = v_in[i];
            float v_diff = v - v_thresh;
            if (v_diff > 0.0f) {
                float i_ca = g_T * (E_Ca - v);
                burst_act[i] = (v_diff + i_ca * 0.01f) * swr_amp;
            } else {
                burst_act[i] = 0.0f;
            }
        }
        // 2. Broadcast to mPFC
        for (size_t i = 0; i < N_MPFC; ++i) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t j = 0; j < N_SUB; j += 8) {
                __m256 w = _mm256_loadu_ps(&W_mpfc[i * N_SUB + j]);
                __m256 b = _mm256_loadu_ps(&burst_act[j]);
                acc = _mm256_fmadd_ps(w, b, acc);
            }
            alignas(32) float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
            y_mpfc[i] = sum > 0.0f ? sum : 0.0f;
        }
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < ITERS; ++it) {
        // 1. Compute burst activation
        for (size_t i = 0; i < N_SUB; ++i) {
            float v = v_in[i];
            float v_diff = v - v_thresh;
            if (v_diff > 0.0f) {
                float i_ca = g_T * (E_Ca - v);
                burst_act[i] = (v_diff + i_ca * 0.01f) * swr_amp;
            } else {
                burst_act[i] = 0.0f;
            }
        }
        // 2. Broadcast to mPFC
        for (size_t i = 0; i < N_MPFC; ++i) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t j = 0; j < N_SUB; j += 8) {
                __m256 w = _mm256_loadu_ps(&W_mpfc[i * N_SUB + j]);
                __m256 b = _mm256_loadu_ps(&burst_act[j]);
                acc = _mm256_fmadd_ps(w, b, acc);
            }
            alignas(32) float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
            y_mpfc[i] = sum > 0.0f ? sum : 0.0f;
        }
        // 3. Broadcast to deep EC
        for (size_t i = 0; i < N_EC_DEEP; ++i) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t j = 0; j < N_SUB; j += 8) {
                __m256 w = _mm256_loadu_ps(&W_ec[i * N_SUB + j]);
                __m256 b = _mm256_loadu_ps(&burst_act[j]);
                acc = _mm256_fmadd_ps(w, b, acc);
            }
            alignas(32) float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
            y_ec[i] = sum > 0.0f ? sum : 0.0f;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    size_t total_synaptic_ops = ITERS * N_SUB * (N_MPFC + N_EC_DEEP) * 2;
    double tflops = (double)total_synaptic_ops / elapsed_sec / 1e12;
    double latency_us = (elapsed_sec / ITERS) * 1e6;

    std::cout << "Elapsed Time: " << elapsed_sec << " s" << std::endl;
    std::cout << "Subicular Broadcasting FMA Rate: " << tflops << " TFLOPS (AVX2-FMA SIMD)" << std::endl;
    std::cout << "SWR Broadcast Cycle Latency: " << latency_us << " us/event" << std::endl;

    return 0;
}
