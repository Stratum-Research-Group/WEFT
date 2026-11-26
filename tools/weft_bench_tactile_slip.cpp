#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

// WEFT AVX2 Microbenchmark: Tactile Slip & Transcortical Grip Escalation
// Benchmarks 32,768 cutaneous fingertip mechanoreceptive afferent units.

void benchmark_tactile_slip_avx2(size_t n, size_t iterations) {
    std::vector<float> f_grip(n);
    std::vector<float> f_load(n);
    std::vector<float> mu(n);
    std::vector<float> fa1_out(n, 0.0f);
    std::vector<float> fa2_out(n, 0.0f);
    std::vector<float> slip_out(n, 0.0f);
    std::vector<float> adapted_grip_out(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        f_grip[i] = 12.0f + 5.0f * (static_cast<float>(i) / static_cast<float>(n));
        f_load[i] = 2.0f + 6.0f * (static_cast<float>(i) / static_cast<float>(n));
        mu[i] = 0.35f;
    }

    const __m256 v_fbase1 = _mm256_set1_ps(15.0f);
    const __m256 v_kfa1 = _mm256_set1_ps(120.0f);
    const __m256 v_max_fa1 = _mm256_set1_ps(350.0f);
    const __m256 v_fbase2 = _mm256_set1_ps(25.0f);
    const __m256 v_kfa2 = _mm256_set1_ps(180.0f);
    const __m256 v_max_fa2 = _mm256_set1_ps(450.0f);
    const __m256 v_safety = _mm256_set1_ps(1.30f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_one = _mm256_set1_ps(1.0f);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < n; i += 8) {
            __m256 vg = _mm256_loadu_ps(&f_grip[i]);
            __m256 vl = _mm256_loadu_ps(&f_load[i]);
            __m256 vmu = _mm256_loadu_ps(&mu[i]);

            // Friction limit: mu * f_grip
            __m256 f_limit = _mm256_mul_ps(vmu, vg);

            // Slip condition: f_load > f_limit
            __m256 is_slip_mask = _mm256_cmp_ps(vl, f_limit, _CMP_GT_OQ);
            __m256 is_slip = _mm256_blendv_ps(v_zero, v_one, is_slip_mask);

            __m256 excess = _mm256_max_ps(v_zero, _mm256_sub_ps(vl, f_limit));

            // FA1 Meissner rate: min(350.0, 15.0 + is_slip * 120.0 * excess)
            __m256 fa1 = _mm256_add_ps(v_fbase1, _mm256_mul_ps(is_slip, _mm256_mul_ps(v_kfa1, excess)));
            fa1 = _mm256_min_ps(fa1, v_max_fa1);

            // FA2 Pacinian rate: min(450.0, 25.0 + is_slip * 180.0 * excess)
            __m256 fa2 = _mm256_add_ps(v_fbase2, _mm256_mul_ps(is_slip, _mm256_mul_ps(v_kfa2, excess)));
            fa2 = _mm256_min_ps(fa2, v_max_fa2);

            // Desired grip force: (1.30 * f_load) / mu
            __m256 desired_grip = _mm256_div_ps(_mm256_mul_ps(v_safety, vl), vmu);
            __m256 delta_grip = _mm256_max_ps(v_zero, _mm256_sub_ps(desired_grip, vg));
            __m256 escalation = _mm256_mul_ps(is_slip, delta_grip);
            __m256 final_grip = _mm256_add_ps(vg, escalation);

            _mm256_storeu_ps(&fa1_out[i], fa1);
            _mm256_storeu_ps(&fa2_out[i], fa2);
            _mm256_storeu_ps(&slip_out[i], is_slip);
            _mm256_storeu_ps(&adapted_grip_out[i], final_grip);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;

    std::cout << "[WEFT AVX2 BENCH] Cutaneous Mechanoreceptor Units: " << n << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Total Iterations: " << iterations << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Total Elapsed Time: " << elapsed.count() << " us" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Time per Iteration: " << (elapsed.count() / iterations) << " us" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Throughput: " 
              << (static_cast<double>(n * iterations) / (elapsed.count() * 1e-6)) / 1e6 << " MAfferents/sec" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Sample Target [n/2]: Load=" << f_load[n/2] << " N, Grip=" 
              << f_grip[n/2] << " N => Slip=" << (slip_out[n/2] > 0.5f ? "YES" : "NO") 
              << ", FA1=" << fa1_out[n/2] << " Hz, FA2=" << fa2_out[n/2] << " Hz, AdaptedGrip=" 
              << adapted_grip_out[n/2] << " N" << std::endl;
}

int main() {
    benchmark_tactile_slip_avx2(32768, 5000);
    return 0;
}
