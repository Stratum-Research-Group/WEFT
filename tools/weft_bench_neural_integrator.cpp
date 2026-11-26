#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>

// Benchmark for Horizontal Oculomotor Neural Integrator (NPH / MVN)
// Evaluates:
// 1. Pulse-to-step transformation: Step = 60.0 + angle * 4.5
// 2. Viscoelastic orbital drift: v_drift = angle / tau_leak
// 3. Eye position retention under high-throughput population simulation
// Target #380 in WEFT Verification Suite.

void benchmark_scalar_integrator(const std::vector<float>& gaze_angle,
                                 const std::vector<float>& tau_leak,
                                 std::vector<float>& step_firing,
                                 std::vector<float>& drift_vel,
                                 size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float a = gaze_angle[i];
        float tau = tau_leak[i];
        step_firing[i] = 60.0f + a * 4.5f;
        drift_vel[i] = a / tau;
    }
}

void benchmark_avx2_integrator(const float* gaze_angle,
                               const float* tau_leak,
                               float* step_firing,
                               float* drift_vel,
                               size_t n) {
    const __m256 v_60 = _mm256_set1_ps(60.0f);
    const __m256 v_slope = _mm256_set1_ps(4.5f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 a = _mm256_loadu_ps(gaze_angle + i);
        __m256 tau = _mm256_loadu_ps(tau_leak + i);

        // Step: 60.0 + a * 4.5
        __m256 step = _mm256_fmadd_ps(a, v_slope, v_60);
        _mm256_storeu_ps(step_firing + i, step);

        // Drift: a / tau
        __m256 r_tau = _mm256_rcp_ps(tau);
        __m256 drift = _mm256_mul_ps(a, r_tau);
        _mm256_storeu_ps(drift_vel + i, drift);
    }
}

int main() {
    const size_t N = 1000000;
    const int WARMUP = 10;
    const int RUNS = 50;

    std::vector<float> gaze_angle(N, 20.0f);
    std::vector<float> tau_leak(N, 25.0f);
    std::vector<float> step_firing(N, 0.0f);
    std::vector<float> drift_vel(N, 0.0f);

    for (int w = 0; w < WARMUP; ++w) {
        benchmark_avx2_integrator(gaze_angle.data(), tau_leak.data(),
                                  step_firing.data(), drift_vel.data(), N);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        benchmark_avx2_integrator(gaze_angle.data(), tau_leak.data(),
                                  step_firing.data(), drift_vel.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double dur_ms = std::chrono::duration<double, std::milli>(end - start).count() / RUNS;

    std::cout << "--- Horizontal Neural Integrator (NPH / MVN) Microbenchmark ---" << std::endl;
    std::cout << "Elements: " << N << " per run (" << RUNS << " iterations)" << std::endl;
    std::cout << "AVX2+FMA Execution Time: " << std::fixed << std::setprecision(3) << dur_ms << " ms" << std::endl;
    double throughput_melem = (N / (dur_ms / 1000.0)) / 1e6;
    std::cout << "AVX2+FMA Throughput: " << throughput_melem << " Melem/s" << std::endl;
    std::cout << "Sample Step Tonic Firing: " << step_firing[0] << " spikes/s" << std::endl;
    std::cout << "Sample Drift Velocity: " << drift_vel[0] << " deg/s" << std::endl;

    return 0;
}
