#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

// WEFT AVX2 Microbenchmark: Superior Colliculus Motor Burst & Tectospinal Gaze Shift
// Benchmarks 32,768 collicular motor map units and tectospinal cervical projection drive.

void benchmark_tectospinal_avx2(size_t n, size_t iterations) {
    std::vector<float> r(n);
    std::vector<float> theta(n);
    std::vector<float> burst_out(n, 0.0f);
    std::vector<float> eye_vel_out(n, 0.0f);
    std::vector<float> head_vel_out(n, 0.0f);
    std::vector<float> tecto_drive_out(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        r[i] = 2.0f + 40.0f * (static_cast<float>(i) / static_cast<float>(n));
        theta[i] = -0.785f + 1.57f * (static_cast<float>(i) / static_cast<float>(n));
    }

    const __m256 v_fmax = _mm256_set1_ps(750.0f);
    const __m256 v_r0 = _mm256_set1_ps(15.0f);
    const __m256 v_head_th = _mm256_set1_ps(15.0f);
    const __m256 v_k_head = _mm256_set1_ps(14.0f);
    const __m256 v_max_head = _mm256_set1_ps(450.0f);
    const __m256 v_k_tecto = _mm256_set1_ps(0.065f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_veye_max = _mm256_set1_ps(650.0f);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < n; i += 8) {
            __m256 vr = _mm256_loadu_ps(&r[i]);
            
            // Rational / polynomial approximation of exp(-r / r0)
            // exp(-x) ~ 1.0 / (1.0 + x + 0.5 * x^2) for fast SIMD
            __m256 x = _mm256_div_ps(vr, v_r0);
            __m256 denom = _mm256_add_ps(v_one, _mm256_add_ps(x, _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(x, x))));
            __m256 exp_neg = _mm256_div_ps(v_one, denom);
            __m256 burst = _mm256_mul_ps(v_fmax, _mm256_sub_ps(v_one, exp_neg));

            // Eye velocity: veye = 650.0 * (1.0 - 1.0 / (1.0 + r/12.0))
            __m256 x_eye = _mm256_div_ps(vr, _mm256_set1_ps(12.0f));
            __m256 denom_eye = _mm256_add_ps(v_one, x_eye);
            __m256 eye_vel = _mm256_mul_ps(v_veye_max, _mm256_sub_ps(v_one, _mm256_div_ps(v_one, denom_eye)));

            // Head velocity: max(0, (r - 15.0) * 14.0) clamped to 450.0
            __m256 diff_head = _mm256_sub_ps(vr, v_head_th);
            __m256 unclipped_head = _mm256_max_ps(v_zero, _mm256_mul_ps(diff_head, v_k_head));
            __m256 head_vel = _mm256_min_ps(unclipped_head, v_max_head);

            // Tectospinal drive: burst * 0.065
            __m256 tecto = _mm256_mul_ps(burst, v_k_tecto);

            _mm256_storeu_ps(&burst_out[i], burst);
            _mm256_storeu_ps(&eye_vel_out[i], eye_vel);
            _mm256_storeu_ps(&head_vel_out[i], head_vel);
            _mm256_storeu_ps(&tecto_drive_out[i], tecto);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;

    std::cout << "[WEFT AVX2 BENCH] Tectospinal & Collicular Motor Map Units: " << n << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Total Iterations: " << iterations << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Total Elapsed Time: " << elapsed.count() << " us" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Time per Iteration: " << (elapsed.count() / iterations) << " us" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Throughput: " 
              << (static_cast<double>(n * iterations) / (elapsed.count() * 1e-6)) / 1e6 << " MUnits/sec" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Sample Target [n/2]: r=" << r[n/2] << " deg => Burst=" 
              << burst_out[n/2] << " Hz, EyeVel=" << eye_vel_out[n/2] << " deg/s, HeadVel=" 
              << head_vel_out[n/2] << " deg/s, Tecto=" << tecto_drive_out[n/2] << " mV" << std::endl;
}

int main() {
    benchmark_tectospinal_avx2(32768, 5000);
    return 0;
}
