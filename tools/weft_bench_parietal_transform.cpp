#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>

// Benchmark for Posterior Parietal Area 5d / 7a Coordinate Transformation
// Evaluates:
// 1. Planar eye-position gain modulation: f(x, y) = RF_retino * (1 + g_x * eye_x + g_y * eye_y)
// 2. Hand-centered reach vector extraction: dx = (target_retino_x + eye_x) - hand_x
// Target #375 in WEFT Verification Suite.

void benchmark_scalar(const std::vector<float>& retino_x,
                      const std::vector<float>& retino_y,
                      const std::vector<float>& eye_x,
                      const std::vector<float>& eye_y,
                      const std::vector<float>& hand_x,
                      const std::vector<float>& hand_y,
                      std::vector<float>& reach_dx,
                      std::vector<float>& reach_dy,
                      std::vector<float>& area7a_act,
                      size_t n) {
    const float gx = 0.05f;
    const float gy = 0.05f;
    for (size_t i = 0; i < n; ++i) {
        // Area 7a gain field
        float gain = 1.0f + gx * eye_x[i] + gy * eye_y[i];
        if (gain < 0.0f) gain = 0.0f;
        float base_rf = std::exp(-0.01f * (retino_x[i] * retino_x[i] + retino_y[i] * retino_y[i]));
        area7a_act[i] = base_rf * gain;

        // Area 5d hand-centered reach vector
        float head_tx = retino_x[i] + eye_x[i];
        float head_ty = retino_y[i] + eye_y[i];
        reach_dx[i] = head_tx - hand_x[i];
        reach_dy[i] = head_ty - hand_y[i];
    }
}

void benchmark_avx2(const float* retino_x,
                    const float* retino_y,
                    const float* eye_x,
                    const float* eye_y,
                    const float* hand_x,
                    const float* hand_y,
                    float* reach_dx,
                    float* reach_dy,
                    float* area7a_act,
                    size_t n) {
    const __m256 v_gx = _mm256_set1_ps(0.05f);
    const __m256 v_gy = _mm256_set1_ps(0.05f);
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_neg_k = _mm256_set1_ps(-0.01f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 rx = _mm256_loadu_ps(retino_x + i);
        __m256 ry = _mm256_loadu_ps(retino_y + i);
        __m256 ex = _mm256_loadu_ps(eye_x + i);
        __m256 ey = _mm256_loadu_ps(eye_y + i);
        __m256 hx = _mm256_loadu_ps(hand_x + i);
        __m256 hy = _mm256_loadu_ps(hand_y + i);

        // Gain field: 1.0 + gx * eye_x + gy * eye_y
        __m256 gain = _mm256_fmadd_ps(v_gx, ex, v_one);
        gain = _mm256_fmadd_ps(v_gy, ey, gain);
        gain = _mm256_max_ps(gain, v_zero);

        // Retinotopic RF Gaussian approximation: max(0, 1.0 - 0.01*(rx^2 + ry^2))
        __m256 r2 = _mm256_fmadd_ps(rx, rx, _mm256_mul_ps(ry, ry));
        __m256 rf = _mm256_max_ps(_mm256_fmadd_ps(v_neg_k, r2, v_one), v_zero);
        __m256 act = _mm256_mul_ps(rf, gain);
        _mm256_storeu_ps(area7a_act + i, act);

        // Area 5d reach vector: (rx + ex) - hx
        __m256 head_x = _mm256_add_ps(rx, ex);
        __m256 head_y = _mm256_add_ps(ry, ey);
        __m256 dx = _mm256_sub_ps(head_x, hx);
        __m256 dy = _mm256_sub_ps(head_y, hy);

        _mm256_storeu_ps(reach_dx + i, dx);
        _mm256_storeu_ps(reach_dy + i, dy);
    }
}

int main() {
    const size_t N = 1000000;
    const int WARMUP = 10;
    const int RUNS = 50;

    std::vector<float> retino_x(N, 12.0f);
    std::vector<float> retino_y(N, -5.0f);
    std::vector<float> eye_x(N, 8.0f);
    std::vector<float> eye_y(N, 2.0f);
    std::vector<float> hand_x(N, 5.0f);
    std::vector<float> hand_y(N, -3.0f);

    std::vector<float> reach_dx(N, 0.0f);
    std::vector<float> reach_dy(N, 0.0f);
    std::vector<float> area7a_act(N, 0.0f);

    for (int w = 0; w < WARMUP; ++w) {
        benchmark_avx2(retino_x.data(), retino_y.data(), eye_x.data(), eye_y.data(),
                       hand_x.data(), hand_y.data(), reach_dx.data(), reach_dy.data(),
                       area7a_act.data(), N);
    }

    auto start_avx = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        benchmark_avx2(retino_x.data(), retino_y.data(), eye_x.data(), eye_y.data(),
                       hand_x.data(), hand_y.data(), reach_dx.data(), reach_dy.data(),
                       area7a_act.data(), N);
    }
    auto end_avx = std::chrono::high_resolution_clock::now();
    double avx_dur_ms = std::chrono::duration<double, std::milli>(end_avx - start_avx).count() / RUNS;

    std::cout << "--- Posterior Parietal Coordinate Transformation Microbenchmark ---" << std::endl;
    std::cout << "Elements: " << N << " per run (" << RUNS << " iterations)" << std::endl;
    std::cout << "AVX2+FMA Execution Time: " << std::fixed << std::setprecision(3) << avx_dur_ms << " ms" << std::endl;
    double throughput_melem = (N / (avx_dur_ms / 1000.0)) / 1e6;
    std::cout << "AVX2+FMA Throughput: " << throughput_melem << " Melem/s" << std::endl;
    std::cout << "Sample Reach Vector: dx = " << reach_dx[0] << ", dy = " << reach_dy[0] << std::endl;
    std::cout << "Sample Area 7a Activation: " << area7a_act[0] << std::endl;

    return 0;
}
