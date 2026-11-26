// Inferior Olive Electrotonic Gap-Junction Coupling AVX2 SIMD Microbenchmark CLI
// Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
// Evaluates diffusive Laplacian gap-junction coupling (Connexin-36) and complex spike generation.

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <immintrin.h>

int main(int argc, char** argv) {
    const size_t N = 1024; // 1,024 Inferior Olivary neurons
    const size_t WARMUP = 100;
    const size_t ITERS = 2000;

    std::cout << "=== WEFT INFERIOR OLIVE GAP JUNCTION AVX2 BENCHMARK ===" << std::endl;
    std::cout << "Dimensions: N=" << N << " olivary neurons (" << (N * N * sizeof(float)) / (1024 * 1024) << " MB Laplacian)" << std::endl;

    std::vector<float> Laplacian(N * N);
    std::vector<float> v_curr(N);
    std::vector<float> i_error(N);
    std::vector<float> v_next(N);
    std::vector<uint8_t> spike_mask(N);

    std::mt19937 rng(42);
    std::normal_distribution<float> v_dist(-60.0f, 5.0f);
    std::uniform_real_distribution<float> err_dist(0.0f, 2.0f);

    // Symmetric graph Laplacian: L = D - A
    for (size_t i = 0; i < N; ++i) {
        float deg = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            if (i != j && (std::abs((int)i - (int)j) < 32)) {
                Laplacian[i * N + j] = -1.0f / 64.0f;
                deg += 1.0f / 64.0f;
            } else if (i != j) {
                Laplacian[i * N + j] = 0.0f;
            }
        }
        Laplacian[i * N + i] = deg;
        v_curr[i] = v_dist(rng);
        i_error[i] = err_dist(rng);
    }

    const float g_gap = 0.5f;
    const float dt_over_tau = 0.1f;
    const float v_thresh = -45.0f;

    // Warmup
    for (size_t it = 0; it < WARMUP; ++it) {
        for (size_t i = 0; i < N; ++i) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t j = 0; j < N; j += 8) {
                __m256 l = _mm256_loadu_ps(&Laplacian[i * N + j]);
                __m256 v = _mm256_loadu_ps(&v_curr[j]);
                acc = _mm256_fmadd_ps(l, v, acc);
            }
            alignas(32) float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            float diff = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

            float i_gap = - g_gap * diff;
            float v_tent = v_curr[i] + dt_over_tau * (- (v_curr[i] - (-60.0f)) + i_gap + i_error[i]);
            bool is_spike = v_tent > v_thresh;
            spike_mask[i] = is_spike ? 1 : 0;
            v_next[i] = is_spike ? -70.0f : v_tent;
        }
        std::copy(v_next.begin(), v_next.end(), v_curr.begin());
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < ITERS; ++it) {
        for (size_t i = 0; i < N; ++i) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t j = 0; j < N; j += 8) {
                __m256 l = _mm256_loadu_ps(&Laplacian[i * N + j]);
                __m256 v = _mm256_loadu_ps(&v_curr[j]);
                acc = _mm256_fmadd_ps(l, v, acc);
            }
            alignas(32) float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            float diff = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

            float i_gap = - g_gap * diff;
            float v_tent = v_curr[i] + dt_over_tau * (- (v_curr[i] - (-60.0f)) + i_gap + i_error[i]);
            bool is_spike = v_tent > v_thresh;
            spike_mask[i] = is_spike ? 1 : 0;
            v_next[i] = is_spike ? -70.0f : v_tent;
        }
        std::copy(v_next.begin(), v_next.end(), v_curr.begin());
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    size_t fma_ops = ITERS * N * N * 2;
    double tflops = (double)fma_ops / elapsed_sec / 1e12;
    double latency_us = (elapsed_sec / ITERS) * 1e6;

    std::cout << "Elapsed Time: " << elapsed_sec << " s" << std::endl;
    std::cout << "Diffusive Gap-Junction FMA Rate: " << tflops << " TFLOPS (AVX2-FMA SIMD)" << std::endl;
    std::cout << "Olivary Synchronization Latency: " << latency_us << " us/cycle" << std::endl;

    return 0;
}
