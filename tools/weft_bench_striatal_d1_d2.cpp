#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>

// Benchmark for Striatal Dopaminergic D1/D2 Opponent Gating Engine
// Evaluates:
// 1. D1 dSPN "Go" pathway facilitation: d1 = cortex * (1.0 + da * 0.8)
// 2. D2 iSPN "NoGo" pathway suppression: d2 = cortex * max(0.1, 1.0 - da * 0.7)
// 3. Opponent contrast & logistic action probability: p = 1 / (1 + exp(-(d1 - d2)/25))
// Target #381 in WEFT Verification Suite.

void benchmark_scalar_striatal(const std::vector<float>& cortex,
                               const std::vector<float>& da_rpe,
                               std::vector<float>& d1_go,
                               std::vector<float>& d2_nogo,
                               std::vector<float>& action_prob,
                               size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float c = cortex[i];
        float da = da_rpe[i];
        float d1 = c * std::fmax(0.1f, 1.0f + da * 0.8f);
        float d2 = c * std::fmax(0.1f, 1.0f - da * 0.7f);
        d1_go[i] = d1;
        d2_nogo[i] = d2;
        float diff = (d1 - d2) / 25.0f;
        action_prob[i] = 1.0f / (1.0f + std::exp(-diff));
    }
}

void benchmark_avx2_striatal(const float* cortex,
                             const float* da_rpe,
                             float* d1_go,
                             float* d2_nogo,
                             float* action_prob,
                             size_t n) {
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_0_1 = _mm256_set1_ps(0.1f);
    const __m256 v_0_8 = _mm256_set1_ps(0.8f);
    const __m256 v_neg_0_7 = _mm256_set1_ps(-0.7f);
    const __m256 v_inv_25 = _mm256_set1_ps(1.0f / 25.0f);
    const __m256 v_half = _mm256_set1_ps(0.5f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 c = _mm256_loadu_ps(cortex + i);
        __m256 da = _mm256_loadu_ps(da_rpe + i);

        // D1: c * max(0.1, 1.0 + da * 0.8)
        __m256 d1_mod = _mm256_max_ps(v_0_1, _mm256_fmadd_ps(da, v_0_8, v_one));
        __m256 d1 = _mm256_mul_ps(c, d1_mod);
        _mm256_storeu_ps(d1_go + i, d1);

        // D2: c * max(0.1, 1.0 - da * 0.7)
        __m256 d2_mod = _mm256_max_ps(v_0_1, _mm256_fmadd_ps(da, v_neg_0_7, v_one));
        __m256 d2 = _mm256_mul_ps(c, d2_mod);
        _mm256_storeu_ps(d2_nogo + i, d2);

        // Net contrast: (d1 - d2) / 25.0
        __m256 diff = _mm256_mul_ps(_mm256_sub_ps(d1, d2), v_inv_25);

        // Sigmoid rational approximation: 0.5 + 0.5 * (diff / (1.0 + |diff|))
        __m256 abs_diff = _mm256_and_ps(diff, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
        __m256 denom = _mm256_add_ps(v_one, abs_diff);
        __m256 soft = _mm256_fmadd_ps(v_half, _mm256_mul_ps(diff, _mm256_rcp_ps(denom)), v_half);
        _mm256_storeu_ps(action_prob + i, soft);
    }
}

int main() {
    const size_t N = 1000000;
    const int WARMUP = 10;
    const int RUNS = 50;

    std::vector<float> cortex(N, 75.0f);
    std::vector<float> da_rpe(N, 0.4f);
    std::vector<float> d1_go(N, 0.0f);
    std::vector<float> d2_nogo(N, 0.0f);
    std::vector<float> action_prob(N, 0.0f);

    for (int w = 0; w < WARMUP; ++w) {
        benchmark_avx2_striatal(cortex.data(), da_rpe.data(),
                                d1_go.data(), d2_nogo.data(), action_prob.data(), N);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        benchmark_avx2_striatal(cortex.data(), da_rpe.data(),
                                d1_go.data(), d2_nogo.data(), action_prob.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double dur_ms = std::chrono::duration<double, std::milli>(end - start).count() / RUNS;

    std::cout << "--- Striatal D1/D2 Opponent Gating Microbenchmark ---" << std::endl;
    std::cout << "Elements: " << N << " per run (" << RUNS << " iterations)" << std::endl;
    std::cout << "AVX2+FMA Execution Time: " << std::fixed << std::setprecision(3) << dur_ms << " ms" << std::endl;
    double throughput_melem = (N / (dur_ms / 1000.0)) / 1e6;
    std::cout << "AVX2+FMA Throughput: " << throughput_melem << " Melem/s" << std::endl;
    std::cout << "Sample D1 Go Drive: " << d1_go[0] << " %" << std::endl;
    std::cout << "Sample D2 NoGo Drive: " << d2_nogo[0] << " %" << std::endl;
    std::cout << "Sample Action Probability: " << action_prob[0] << std::endl;

    return 0;
}
