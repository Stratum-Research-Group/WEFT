#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

// WEFT AVX2 Microbenchmark: Lateral Corticospinal Tract (LCST) & Fractionated Grasp
// Benchmarks 32,768 corticomotoneuronal (CM) motor units and alpha-motoneuron pools.

void benchmark_corticospinal_avx2(size_t n, size_t iterations) {
    std::vector<float> target_force(n);
    std::vector<float> m1_rate_out(n, 0.0f);
    std::vector<float> alpha_recruit_out(n, 0.0f);
    std::vector<float> active_force_out(n, 0.0f);
    std::vector<float> spillover_out(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        target_force[i] = 0.5f + 25.0f * (static_cast<float>(i) / static_cast<float>(n));
    }

    const __m256 v_fbase = _mm256_set1_ps(12.0f);
    const __m256 v_km1 = _mm256_set1_ps(8.5f);
    const __m256 v_inv60 = _mm256_set1_ps(1.0f / 60.0f);
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_fidelity = _mm256_set1_ps(0.992f);
    const __m256 v_kspill = _mm256_set1_ps(0.035f);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < n; i += 8) {
            __m256 vf = _mm256_loadu_ps(&target_force[i]);

            // M1 firing rate: fbase + km1 * target_force
            __m256 m1_rate = _mm256_add_ps(v_fbase, _mm256_mul_ps(v_km1, vf));

            // Alpha-motoneuron recruitment: 1.0 - 1.0 / (1.0 + m1 / 60.0 + 0.5 * (m1/60.0)^2)
            __m256 x = _mm256_mul_ps(m1_rate, v_inv60);
            __m256 denom = _mm256_add_ps(v_one, _mm256_add_ps(x, _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(x, x))));
            __m256 exp_neg = _mm256_div_ps(v_one, denom);
            __m256 recruit = _mm256_sub_ps(v_one, exp_neg);
            recruit = _mm256_max_ps(v_zero, _mm256_min_ps(v_one, recruit));

            // Active force: vf * 0.992
            __m256 active_f = _mm256_mul_ps(vf, v_fidelity);

            // Spillover: active_f * 0.035
            __m256 spill = _mm256_mul_ps(active_f, v_kspill);

            _mm256_storeu_ps(&m1_rate_out[i], m1_rate);
            _mm256_storeu_ps(&alpha_recruit_out[i], recruit);
            _mm256_storeu_ps(&active_force_out[i], active_f);
            _mm256_storeu_ps(&spillover_out[i], spill);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;

    std::cout << "[WEFT AVX2 BENCH] Corticospinal Motor Units: " << n << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Total Iterations: " << iterations << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Total Elapsed Time: " << elapsed.count() << " us" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Time per Iteration: " << (elapsed.count() / iterations) << " us" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Throughput: " 
              << (static_cast<double>(n * iterations) / (elapsed.count() * 1e-6)) / 1e6 << " MMotorUnits/sec" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Sample Target [n/2]: F=" << target_force[n/2] << " N => M1=" 
              << m1_rate_out[n/2] << " Hz, Recruit=" << alpha_recruit_out[n/2] * 100.0f << "%, ActiveF=" 
              << active_force_out[n/2] << " N, Spillover=" << spillover_out[n/2] << " N" << std::endl;
}

int main() {
    benchmark_corticospinal_avx2(32768, 5000);
    return 0;
}
