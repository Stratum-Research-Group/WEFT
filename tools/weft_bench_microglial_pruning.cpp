// tools/weft_bench_microglial_pruning.cpp
// Native AVX2/FMA SIMD benchmark for Microglial Synaptic Pruning kinetics
// Computes CD47 suppression of C1q/C3b opsonization, CR3 phagocytic engulfment, and spine survival

#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

static constexpr size_t N_CHANNELS = 65536;
static constexpr int N_ITERATIONS = 1000;

void benchmark_microglial_pruning_avx2(
    const float* __restrict__ cd47_levels,
    const float* __restrict__ c1q_stress,
    float* __restrict__ out_c3b,
    float* __restrict__ out_cr3,
    float* __restrict__ out_survival,
    float* __restrict__ out_ei_ratio,
    size_t n
) {
    const __m256 v_hundred = _mm256_set1_ps(100.0f);
    const __m256 v_c1q_scale = _mm256_set1_ps(1.2f);
    const __m256 v_cd47_suppr = _mm256_set1_ps(0.0085f); // 0.85 / 100
    const __m256 v_c3b_min = _mm256_set1_ps(5.0f);
    const __m256 v_cr3_k = _mm256_set1_ps(25.0f);
    const __m256 v_cr3_loss = _mm256_set1_ps(0.85f);
    const __m256 v_survival_min = _mm256_set1_ps(10.0f);
    const __m256 v_ei_scale = _mm256_set1_ps(1.15f / 85.0f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_cd47 = _mm256_loadu_ps(cd47_levels + i);
        __m256 v_c1q = _mm256_loadu_ps(c1q_stress + i);

        // suppr = cd47 * 0.0085
        __m256 v_suppr = _mm256_mul_ps(v_cd47, v_cd47_suppr);
        // c3b_raw = (c1q * 1.2) * (1.0 - suppr)
        __m256 v_c1q_prod = _mm256_mul_ps(v_c1q, v_c1q_scale);
        __m256 v_one_minus_suppr = _mm256_sub_ps(_mm256_set1_ps(1.0f), v_suppr);
        __m256 v_c3b_raw = _mm256_mul_ps(v_c1q_prod, v_one_minus_suppr);
        __m256 v_c3b = _mm256_max_ps(v_c3b_min, _mm256_min_ps(v_hundred, v_c3b_raw));
        _mm256_storeu_ps(out_c3b + i, v_c3b);

        // cr3 = (c3b / (c3b + 25.0)) * 100.0
        __m256 v_cr3_denom = _mm256_add_ps(v_c3b, v_cr3_k);
        __m256 v_cr3 = _mm256_mul_ps(_mm256_div_ps(v_c3b, v_cr3_denom), v_hundred);
        _mm256_storeu_ps(out_cr3 + i, v_cr3);

        // survival = max(10.0, 100.0 - cr3 * 0.85)
        __m256 v_loss = _mm256_mul_ps(v_cr3, v_cr3_loss);
        __m256 v_survival = _mm256_max_ps(v_survival_min, _mm256_sub_ps(v_hundred, v_loss));
        _mm256_storeu_ps(out_survival + i, v_survival);

        // ei_ratio = 1.15 * (survival / 85.0)
        __m256 v_ei = _mm256_mul_ps(v_survival, v_ei_scale);
        _mm256_storeu_ps(out_ei_ratio + i, v_ei);
    }
}

int main() {
    std::vector<float> cd47(N_CHANNELS, 75.0f);
    std::vector<float> c1q(N_CHANNELS, 35.0f);
    std::vector<float> c3b(N_CHANNELS);
    std::vector<float> cr3(N_CHANNELS);
    std::vector<float> survival(N_CHANNELS);
    std::vector<float> ei_ratio(N_CHANNELS);

    // Warmup
    for (int i = 0; i < 50; ++i) {
        benchmark_microglial_pruning_avx2(
            cd47.data(), c1q.data(),
            c3b.data(), cr3.data(), survival.data(), ei_ratio.data(),
            N_CHANNELS
        );
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERATIONS; ++i) {
        benchmark_microglial_pruning_avx2(
            cd47.data(), c1q.data(),
            c3b.data(), cr3.data(), survival.data(), ei_ratio.data(),
            N_CHANNELS
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double avg_us = total_us / N_ITERATIONS;
    double throughput_mch_s = (static_cast<double>(N_CHANNELS) / (avg_us * 1e-6)) / 1e6;

    std::cout << "[AVX2-FMA Benchmark] Microglial Synaptic Pruning Kinetics" << std::endl;
    std::cout << "Channels: " << N_CHANNELS << " | Iterations: " << N_ITERATIONS << std::endl;
    std::cout << "Avg latency per iteration: " << avg_us << " us" << std::endl;
    std::cout << "Throughput: " << throughput_mch_s << " Million channels/sec" << std::endl;
    std::cout << "Sample verification [Channel 0]:" << std::endl;
    std::cout << "  CD47: " << cd47[0] << "% | C1q Stress: " << c1q[0] << std::endl;
    std::cout << "  C3b Opsonization: " << c3b[0] << "% | CR3 Phagocytosis: " << cr3[0] << "%" << std::endl;
    std::cout << "  Spine Survival: " << survival[0] << "% | E/I Ratio: " << ei_ratio[0] << std::endl;

    return 0;
}
