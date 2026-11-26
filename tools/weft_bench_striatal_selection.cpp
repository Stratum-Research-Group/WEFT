// Striatal D1/D2 Opponent Action Selection AVX2 SIMD Microbenchmark CLI
// Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
// Evaluates direct (D1) and indirect (D2) corticostriatal projection throughput and GPi gating latency.

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <immintrin.h>

int main(int argc, char** argv) {
    const size_t N_CTX = 1024; // 1,024 Cortical input features
    const size_t N_ACT = 64;   // 64 Candidate motor/cognitive actions
    const size_t WARMUP = 100;
    const size_t ITERS = 5000;

    std::cout << "=== WEFT STRIATAL D1/D2 ACTION SELECTION AVX2 BENCHMARK ===" << std::endl;
    std::cout << "Dimensions: N_CTX=" << N_CTX << ", N_ACT=" << N_ACT << std::endl;

    std::vector<float> W_d1(N_ACT * N_CTX);
    std::vector<float> W_d2(N_ACT * N_CTX);
    std::vector<float> x_ctx(N_CTX);
    std::vector<float> y_d1(N_ACT);
    std::vector<float> y_d2(N_ACT);
    std::vector<uint8_t> action_gated(N_ACT);

    std::mt19937 rng(42);
    std::normal_distribution<float> w_dist(0.0f, 1.0f / std::sqrt(N_CTX));
    std::uniform_real_distribution<float> act_dist(0.0f, 1.0f);

    for (size_t i = 0; i < N_ACT * N_CTX; ++i) {
        W_d1[i] = w_dist(rng);
        W_d2[i] = w_dist(rng);
    }
    for (size_t i = 0; i < N_CTX; ++i) x_ctx[i] = act_dist(rng);

    const float delta_da = 0.65f; // Positive dopamine burst
    const float beta_d1 = 0.8f;
    const float beta_d2 = 0.8f;
    const float mod_d1 = std::max(0.0f, 1.0f + beta_d1 * delta_da);
    const float mod_d2 = std::max(0.0f, 1.0f - beta_d2 * delta_da);
    const float gpi_thresh = 0.2f;

    // Warmup
    for (size_t it = 0; it < WARMUP; ++it) {
        for (size_t a = 0; a < N_ACT; ++a) {
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            for (size_t k = 0; k < N_CTX; k += 8) {
                __m256 x = _mm256_loadu_ps(&x_ctx[k]);
                __m256 w1 = _mm256_loadu_ps(&W_d1[a * N_CTX + k]);
                __m256 w2 = _mm256_loadu_ps(&W_d2[a * N_CTX + k]);
                acc1 = _mm256_fmadd_ps(w1, x, acc1);
                acc2 = _mm256_fmadd_ps(w2, x, acc2);
            }
            alignas(32) float tmp1[8], tmp2[8];
            _mm256_storeu_ps(tmp1, acc1);
            _mm256_storeu_ps(tmp2, acc2);
            float sum1 = tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3] + tmp1[4] + tmp1[5] + tmp1[6] + tmp1[7];
            float sum2 = tmp2[0] + tmp2[1] + tmp2[2] + tmp2[3] + tmp2[4] + tmp2[5] + tmp2[6] + tmp2[7];

            float d1 = std::max(0.0f, sum1 * mod_d1);
            float d2 = std::max(0.0f, sum2 * mod_d2);
            y_d1[a] = d1;
            y_d2[a] = d2;
            action_gated[a] = (d1 > d2 + gpi_thresh) ? 1 : 0;
        }
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < ITERS; ++it) {
        for (size_t a = 0; a < N_ACT; ++a) {
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            for (size_t k = 0; k < N_CTX; k += 8) {
                __m256 x = _mm256_loadu_ps(&x_ctx[k]);
                __m256 w1 = _mm256_loadu_ps(&W_d1[a * N_CTX + k]);
                __m256 w2 = _mm256_loadu_ps(&W_d2[a * N_CTX + k]);
                acc1 = _mm256_fmadd_ps(w1, x, acc1);
                acc2 = _mm256_fmadd_ps(w2, x, acc2);
            }
            alignas(32) float tmp1[8], tmp2[8];
            _mm256_storeu_ps(tmp1, acc1);
            _mm256_storeu_ps(tmp2, acc2);
            float sum1 = tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3] + tmp1[4] + tmp1[5] + tmp1[6] + tmp1[7];
            float sum2 = tmp2[0] + tmp2[1] + tmp2[2] + tmp2[3] + tmp2[4] + tmp2[5] + tmp2[6] + tmp2[7];

            float d1 = std::max(0.0f, sum1 * mod_d1);
            float d2 = std::max(0.0f, sum2 * mod_d2);
            y_d1[a] = d1;
            y_d2[a] = d2;
            action_gated[a] = (d1 > d2 + gpi_thresh) ? 1 : 0;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    size_t fma_ops = ITERS * N_ACT * N_CTX * 4; // 2 matrix-vector mults
    double tflops = (double)fma_ops / elapsed_sec / 1e12;
    double latency_us = (elapsed_sec / ITERS) * 1e6;

    std::cout << "Elapsed Time: " << elapsed_sec << " s" << std::endl;
    std::cout << "Striatal D1/D2 Opponent FMA Rate: " << tflops << " TFLOPS (AVX2-FMA SIMD)" << std::endl;
    std::cout << "Action Selection Cycle Latency: " << latency_us << " us/cycle" << std::endl;

    return 0;
}
