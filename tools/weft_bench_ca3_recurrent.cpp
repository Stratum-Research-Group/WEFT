// Hippocampal CA3 Recurrent Collateral SIMD AVX2 Microbenchmark CLI
// Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
// Evaluates dense and sparse auto-associative recurrent collateral settling in CA3 networks.

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <immintrin.h>

static inline float fast_tanh_approx(float x) {
    // Fast polynomial tanh approximation
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + 28.0f * x2));
    float res = a / b;
    return (x > 4.0f) ? 1.0f : (x < -4.0f) ? -1.0f : res;
}

int main(int argc, char** argv) {
    const size_t N = 1024; // 1,024 CA3 Pyramidal Neurons
    const size_t STEPS = 20; // 20 recurrent iterations
    const size_t WARMUP = 100;
    const size_t ITERS = 1000;
    const float beta = 0.35f;

    std::cout << "=== WEFT CA3 RECURRENT COLLATERAL AVX2 BENCHMARK ===" << std::endl;
    std::cout << "Network Dimension: N=" << N << " neurons (" << (N * N * sizeof(float)) / (1024 * 1024) << " MB weights)" << std::endl;
    std::cout << "Recurrent Steps per Episode: " << STEPS << std::endl;

    // Allocate aligned memory for SIMD AVX2
    std::vector<float> W_rc(N * N);
    std::vector<float> x_state(N);
    std::vector<float> x0(N);
    std::vector<float> x_next(N);

    std::mt19937 rng(42);
    std::normal_distribution<float> w_dist(0.0f, 1.0f / std::sqrt(N));
    std::uniform_real_distribution<float> bin_dist(-1.0f, 1.0f);

    for (size_t i = 0; i < N; ++i) {
        x0[i] = bin_dist(rng) > 0.0f ? 1.0f : -1.0f;
        x_state[i] = x0[i];
    }
    // Occlude 40% of the cue
    for (size_t i = 0; i < N * 4 / 10; ++i) {
        x_state[i] = 0.0f;
    }

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            W_rc[i * N + j] = w_dist(rng);
        }
    }

    // Warmup
    for (size_t it = 0; it < WARMUP; ++it) {
        for (size_t s = 0; s < STEPS; ++s) {
            for (size_t i = 0; i < N; ++i) {
                __m256 acc = _mm256_setzero_ps();
                for (size_t j = 0; j < N; j += 8) {
                    __m256 w_vec = _mm256_loadu_ps(&W_rc[i * N + j]);
                    __m256 x_vec = _mm256_loadu_ps(&x_state[j]);
                    acc = _mm256_fmadd_ps(w_vec, x_vec, acc);
                }
                // Horizontal add
                alignas(32) float tmp[8];
                _mm256_storeu_ps(tmp, acc);
                float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
                x_next[i] = fast_tanh_approx(sum + beta * x0[i]);
            }
            std::copy(x_next.begin(), x_next.end(), x_state.begin());
        }
    }

    // Timed benchmark
    auto start = std::chrono::high_resolution_clock::now();
    size_t total_recurrent_updates = 0;

    for (size_t it = 0; it < ITERS; ++it) {
        for (size_t s = 0; s < STEPS; ++s) {
            for (size_t i = 0; i < N; ++i) {
                __m256 acc = _mm256_setzero_ps();
                for (size_t j = 0; j < N; j += 8) {
                    __m256 w_vec = _mm256_loadu_ps(&W_rc[i * N + j]);
                    __m256 x_vec = _mm256_loadu_ps(&x_state[j]);
                    acc = _mm256_fmadd_ps(w_vec, x_vec, acc);
                }
                alignas(32) float tmp[8];
                _mm256_storeu_ps(tmp, acc);
                float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
                x_next[i] = fast_tanh_approx(sum + beta * x0[i]);
            }
            std::copy(x_next.begin(), x_next.end(), x_state.begin());
            total_recurrent_updates += N;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    double throughput = (double)total_recurrent_updates / elapsed_sec / 1e9; // Billion updates/sec
    double syn_ops = (double)total_recurrent_updates * N * 2.0 / elapsed_sec / 1e12; // TFLOPs equivalent

    std::cout << "Elapsed Time: " << elapsed_sec << " s" << std::endl;
    std::cout << "Recurrent Update Throughput: " << throughput << " Billion neuron-updates/s" << std::endl;
    std::cout << "Synaptic FMA Rate: " << syn_ops << " TFLOPS (AVX2-FMA SIMD)" << std::endl;
    std::cout << "Attractor Settling Latency: " << (elapsed_sec / (ITERS * STEPS)) * 1e6 << " us/step" << std::endl;

    return 0;
}
