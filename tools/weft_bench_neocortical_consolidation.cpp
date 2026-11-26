// Neocortical Systems Consolidation AVX2 SIMD Microbenchmark CLI
// Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
// Evaluates slow interleaved cortical plasticity updates and multi-layer associative weight drift.

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <immintrin.h>

int main(int argc, char** argv) {
    const size_t M = 1024; // 1,024 Neocortical neurons
    const size_t K = 512;  // 512 Subicular/Hippocampal input dimensions
    const size_t B = 16;   // 16 SWR replay episodes per batch
    const size_t WARMUP = 50;
    const size_t ITERS = 1000;

    std::cout << "=== WEFT NEOCORTICAL SYSTEMS CONSOLIDATION AVX2 BENCHMARK ===" << std::endl;
    std::cout << "Dimensions: M=" << M << ", K=" << K << ", Batch=" << B << std::endl;

    std::vector<float> W_cortex(M * K);
    std::vector<float> X_replay(B * K);
    std::vector<float> Y_cortex(B * M);
    std::vector<float> Delta_W(M * K);

    std::mt19937 rng(42);
    std::normal_distribution<float> w_dist(0.0f, 1.0f / std::sqrt(K));
    std::uniform_real_distribution<float> act_dist(-1.0f, 1.0f);

    for (size_t i = 0; i < M * K; ++i) W_cortex[i] = w_dist(rng);
    for (size_t i = 0; i < B * K; ++i) X_replay[i] = act_dist(rng);

    const float eta_cortex = 0.005f;
    const float decay = 1e-4f;

    // Warmup
    for (size_t it = 0; it < WARMUP; ++it) {
        // Forward pass: Y = ReLU(X @ W^T)
        for (size_t b = 0; b < B; ++b) {
            for (size_t m = 0; m < M; ++m) {
                __m256 acc = _mm256_setzero_ps();
                for (size_t k = 0; k < K; k += 8) {
                    __m256 x = _mm256_loadu_ps(&X_replay[b * K + k]);
                    __m256 w = _mm256_loadu_ps(&W_cortex[m * K + k]);
                    acc = _mm256_fmadd_ps(x, w, acc);
                }
                alignas(32) float tmp[8];
                _mm256_storeu_ps(tmp, acc);
                float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
                Y_cortex[b * M + m] = sum > 0.0f ? sum : 0.0f;
            }
        }
        // Interleaved update: Delta_W = eta * (Y^T @ X) - eta * decay * W
        for (size_t m = 0; m < M; ++m) {
            for (size_t k = 0; k < K; k += 8) {
                __m256 acc = _mm256_setzero_ps();
                for (size_t b = 0; b < B; ++b) {
                    __m256 y_val = _mm256_set1_ps(Y_cortex[b * M + m]);
                    __m256 x_vec = _mm256_loadu_ps(&X_replay[b * K + k]);
                    acc = _mm256_fmadd_ps(y_val, x_vec, acc);
                }
                __m256 w_vec = _mm256_loadu_ps(&W_cortex[m * K + k]);
                __m256 delta = _mm256_sub_ps(
                    _mm256_mul_ps(_mm256_set1_ps(eta_cortex), acc),
                    _mm256_mul_ps(_mm256_set1_ps(eta_cortex * decay), w_vec)
                );
                _mm256_storeu_ps(&Delta_W[m * K + k], delta);
            }
        }
    }

    // Timed benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < ITERS; ++it) {
        // Forward pass: Y = ReLU(X @ W^T)
        for (size_t b = 0; b < B; ++b) {
            for (size_t m = 0; m < M; ++m) {
                __m256 acc = _mm256_setzero_ps();
                for (size_t k = 0; k < K; k += 8) {
                    __m256 x = _mm256_loadu_ps(&X_replay[b * K + k]);
                    __m256 w = _mm256_loadu_ps(&W_cortex[m * K + k]);
                    acc = _mm256_fmadd_ps(x, w, acc);
                }
                alignas(32) float tmp[8];
                _mm256_storeu_ps(tmp, acc);
                float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
                Y_cortex[b * M + m] = sum > 0.0f ? sum : 0.0f;
            }
        }
        // Interleaved update: Delta_W = eta * (Y^T @ X) - eta * decay * W
        for (size_t m = 0; m < M; ++m) {
            for (size_t k = 0; k < K; k += 8) {
                __m256 acc = _mm256_setzero_ps();
                for (size_t b = 0; b < B; ++b) {
                    __m256 y_val = _mm256_set1_ps(Y_cortex[b * M + m]);
                    __m256 x_vec = _mm256_loadu_ps(&X_replay[b * K + k]);
                    acc = _mm256_fmadd_ps(y_val, x_vec, acc);
                }
                __m256 w_vec = _mm256_loadu_ps(&W_cortex[m * K + k]);
                __m256 delta = _mm256_sub_ps(
                    _mm256_mul_ps(_mm256_set1_ps(eta_cortex), acc),
                    _mm256_mul_ps(_mm256_set1_ps(eta_cortex * decay), w_vec)
                );
                _mm256_storeu_ps(&Delta_W[m * K + k], delta);
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    size_t fma_ops = ITERS * (B * M * K * 2 + M * K * B * 2);
    double tflops = (double)fma_ops / elapsed_sec / 1e12;
    double step_latency_us = (elapsed_sec / ITERS) * 1e6;

    std::cout << "Elapsed Time: " << elapsed_sec << " s" << std::endl;
    std::cout << "Interleaved Consolidation FMA Rate: " << tflops << " TFLOPS (AVX2-FMA SIMD)" << std::endl;
    std::cout << "Consolidation Epoch Latency: " << step_latency_us << " us/batch" << std::endl;

    return 0;
}
