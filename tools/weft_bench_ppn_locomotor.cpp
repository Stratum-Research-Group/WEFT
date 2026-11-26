// Pedunculopontine Nucleus (PPN) Locomotor Command AVX2 SIMD Microbenchmark CLI
// Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
// Evaluates PPN disinhibition, cholinergic/glutamatergic activation, and CPG frequency scaling.

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <immintrin.h>

int main(int argc, char** argv) {
    const size_t N_PPN = 512; // 512 PPN projection neurons
    const size_t N_GPI = 256; // 256 GPi descending inhibitory afferents
    const size_t WARMUP = 100;
    const size_t ITERS = 5000;

    std::cout << "=== WEFT PPN LOCOMOTOR GATING AVX2 BENCHMARK ===" << std::endl;
    std::cout << "Dimensions: N_PPN=" << N_PPN << ", N_GPI=" << N_GPI << std::endl;

    std::vector<float> W_gpi(N_PPN * N_GPI);
    std::vector<float> x_gpi(N_GPI);
    std::vector<float> i_drive(N_PPN);
    std::vector<float> y_glu(N_PPN);
    std::vector<float> y_ach(N_PPN);

    std::mt19937 rng(42);
    std::normal_distribution<float> w_dist(0.0f, 1.0f / std::sqrt(N_GPI));
    std::uniform_real_distribution<float> act_dist(0.0f, 1.0f);

    for (size_t i = 0; i < N_PPN * N_GPI; ++i) W_gpi[i] = w_dist(rng);
    for (size_t i = 0; i < N_GPI; ++i) x_gpi[i] = act_dist(rng);
    for (size_t i = 0; i < N_PPN; ++i) i_drive[i] = act_dist(rng) * 1.5f;

    const float ach_gain = 0.85f;
    const float theta_loco = 0.2f;
    const float alpha_step = 0.08f;

    // Warmup
    for (size_t it = 0; it < WARMUP; ++it) {
        float mean_glu_excess = 0.0f;
        for (size_t p = 0; p < N_PPN; ++p) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t k = 0; k < N_GPI; k += 8) {
                __m256 w = _mm256_loadu_ps(&W_gpi[p * N_GPI + k]);
                __m256 x = _mm256_loadu_ps(&x_gpi[k]);
                acc = _mm256_fmadd_ps(w, x, acc);
            }
            alignas(32) float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            float gaba = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

            float net = i_drive[p] - gaba;
            float glu = net > 0.0f ? net : 0.0f;
            float ach = glu * ach_gain;
            y_glu[p] = glu;
            y_ach[p] = ach;

            float excess = glu - theta_loco;
            if (excess > 0.0f) mean_glu_excess += excess;
        }
        float cpg_step_hz = alpha_step * (mean_glu_excess / N_PPN);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < ITERS; ++it) {
        float mean_glu_excess = 0.0f;
        for (size_t p = 0; p < N_PPN; ++p) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t k = 0; k < N_GPI; k += 8) {
                __m256 w = _mm256_loadu_ps(&W_gpi[p * N_GPI + k]);
                __m256 x = _mm256_loadu_ps(&x_gpi[k]);
                acc = _mm256_fmadd_ps(w, x, acc);
            }
            alignas(32) float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            float gaba = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

            float net = i_drive[p] - gaba;
            float glu = net > 0.0f ? net : 0.0f;
            float ach = glu * ach_gain;
            y_glu[p] = glu;
            y_ach[p] = ach;

            float excess = glu - theta_loco;
            if (excess > 0.0f) mean_glu_excess += excess;
        }
        float cpg_step_hz = alpha_step * (mean_glu_excess / N_PPN);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    size_t fma_ops = ITERS * N_PPN * N_GPI * 2;
    double tflops = (double)fma_ops / elapsed_sec / 1e12;
    double latency_us = (elapsed_sec / ITERS) * 1e6;

    std::cout << "Elapsed Time: " << elapsed_sec << " s" << std::endl;
    std::cout << "PPN Disinhibition FMA Rate: " << tflops << " TFLOPS (AVX2-FMA SIMD)" << std::endl;
    std::cout << "Locomotor Command Cycle Latency: " << latency_us << " us/cycle" << std::endl;

    return 0;
}
