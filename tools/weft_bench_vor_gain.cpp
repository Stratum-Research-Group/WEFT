#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>

// Benchmark for Vestibulo-Ocular Reflex (VOR) Three-Neuron Arc & Retinal Slip Processor
// Evaluates:
// 1. Semicircular canal angular velocity transduction
// 2. Compensatory counter-rotational eye velocity: v_eye = -gain * v_head
// 3. Residual retinal slip velocity: slip = |v_head + v_eye|
// Target #377 in WEFT Verification Suite.

void benchmark_scalar_vor(const std::vector<float>& head_vel,
                          const std::vector<float>& vor_gain,
                          std::vector<float>& eye_vel,
                          std::vector<float>& retinal_slip,
                          size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float h = head_vel[i];
        float g = vor_gain[i];
        float e = -h * g;
        eye_vel[i] = e;
        retinal_slip[i] = std::fabs(h + e);
    }
}

void benchmark_avx2_vor(const float* head_vel,
                        const float* vor_gain,
                        float* eye_vel,
                        float* retinal_slip,
                        size_t n) {
    const __m256 v_neg_one = _mm256_set1_ps(-1.0f);
    const __m256 v_sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    for (size_t i = 0; i < n; i += 8) {
        __m256 h = _mm256_loadu_ps(head_vel + i);
        __m256 g = _mm256_loadu_ps(vor_gain + i);

        // Eye velocity: -h * g
        __m256 e = _mm256_mul_ps(_mm256_mul_ps(h, g), v_neg_one);
        _mm256_storeu_ps(eye_vel + i, e);

        // Retinal slip: |h + e|
        __m256 slip = _mm256_add_ps(h, e);
        __m256 abs_slip = _mm256_and_ps(slip, v_sign_mask);
        _mm256_storeu_ps(retinal_slip + i, abs_slip);
    }
}

int main() {
    const size_t N = 1000000;
    const int WARMUP = 10;
    const int RUNS = 50;

    std::vector<float> head_vel(N, 120.0f);
    std::vector<float> vor_gain(N, 0.98f);
    std::vector<float> eye_vel(N, 0.0f);
    std::vector<float> retinal_slip(N, 0.0f);

    for (int w = 0; w < WARMUP; ++w) {
        benchmark_avx2_vor(head_vel.data(), vor_gain.data(), eye_vel.data(), retinal_slip.data(), N);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        benchmark_avx2_vor(head_vel.data(), vor_gain.data(), eye_vel.data(), retinal_slip.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double dur_ms = std::chrono::duration<double, std::milli>(end - start).count() / RUNS;

    std::cout << "--- Vestibulo-Ocular Reflex (VOR) Retinal Slip Microbenchmark ---" << std::endl;
    std::cout << "Elements: " << N << " per run (" << RUNS << " iterations)" << std::endl;
    std::cout << "AVX2+FMA Execution Time: " << std::fixed << std::setprecision(3) << dur_ms << " ms" << std::endl;
    double throughput_melem = (N / (dur_ms / 1000.0)) / 1e6;
    std::cout << "AVX2+FMA Throughput: " << throughput_melem << " Melem/s" << std::endl;
    std::cout << "Sample Eye Velocity: " << eye_vel[0] << " deg/s" << std::endl;
    std::cout << "Sample Retinal Slip: " << retinal_slip[0] << " deg/s" << std::endl;

    return 0;
}
