#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>

// Target #427: Hypothalamic AgRP/POMC Metabolic Energy Homeostasis Invariants
// SIMD AVX2/FMA Vectorized Benchmark: 65,536 parallel hypothalamic arcuate circuits.

constexpr size_t N_CHANNELS = 65536;
constexpr size_t ITERATIONS = 1000;

void benchmark_avx2_agrp_pomc(
    const float* ghrelin,
    const float* leptin,
    float* agrp_out,
    float* pomc_out,
    float* mc4r_out,
    float* ee_out,
    size_t n
) {
    const __m256 v_c38 = _mm256_set1_ps(0.38f);
    const __m256 v_c15 = _mm256_set1_ps(0.15f);
    const __m256 v_c5  = _mm256_set1_ps(5.0f);
    const __m256 v_c28 = _mm256_set1_ps(0.28f);
    const __m256 v_c35 = _mm256_set1_ps(0.35f);
    const __m256 v_c3  = _mm256_set1_ps(3.0f);
    const __m256 v_c42 = _mm256_set1_ps(4.2f);
    const __m256 v_c18 = _mm256_set1_ps(1.8f);
    const __m256 v_c30 = _mm256_set1_ps(30.0f);
    const __m256 v_c85 = _mm256_set1_ps(0.85f);
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_max = _mm256_set1_ps(100.0f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 g = _mm256_loadu_ps(ghrelin + i);
        __m256 l = _mm256_loadu_ps(leptin + i);

        // agrp = max(1.0, g * 0.38 - l * 0.15 + 5.0)
        __m256 agrp = _mm256_fmadd_ps(g, v_c38, v_c5);
        agrp = _mm256_fnmadd_ps(l, v_c15, agrp);
        agrp = _mm256_max_ps(agrp, v_one);

        // pomc = max(0.5, l * 0.28 - agrp * 0.35 + 3.0)
        __m256 pomc = _mm256_fmadd_ps(l, v_c28, v_c3);
        pomc = _mm256_fnmadd_ps(agrp, v_c35, pomc);
        pomc = _mm256_max_ps(pomc, _mm256_set1_ps(0.5f));

        // mc4r = clamp(pomc * 4.2 - agrp * 1.8 + 30.0, 5.0, 95.0)
        __m256 mc4r = _mm256_fmadd_ps(pomc, v_c42, v_c30);
        mc4r = _mm256_fnmadd_ps(agrp, v_c18, mc4r);
        mc4r = _mm256_max_ps(_mm256_set1_ps(5.0f), _mm256_min_ps(mc4r, _mm256_set1_ps(95.0f)));

        // ee = mc4r * 0.85 + 5.0
        __m256 ee = _mm256_fmadd_ps(mc4r, v_c85, v_c5);

        _mm256_storeu_ps(agrp_out + i, agrp);
        _mm256_storeu_ps(pomc_out + i, pomc);
        _mm256_storeu_ps(mc4r_out + i, mc4r);
        _mm256_storeu_ps(ee_out + i, ee);
    }
}

int main() {
    std::vector<float> ghrelin(N_CHANNELS, 65.0f);
    std::vector<float> leptin(N_CHANNELS, 35.0f);
    std::vector<float> agrp(N_CHANNELS, 0.0f);
    std::vector<float> pomc(N_CHANNELS, 0.0f);
    std::vector<float> mc4r(N_CHANNELS, 0.0f);
    std::vector<float> ee(N_CHANNELS, 0.0f);

    std::cout << "--- WEFT Hypothalamic AgRP/POMC AVX2/FMA Microbenchmark ---" << std::endl;
    std::cout << "Simulating " << N_CHANNELS << " parallel hypothalamic units across " << ITERATIONS << " steps." << std::endl;

    // Warm-up
    for (int i = 0; i < 50; ++i) {
        benchmark_avx2_agrp_pomc(ghrelin.data(), leptin.data(), agrp.data(), pomc.data(), mc4r.data(), ee.data(), N_CHANNELS);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < ITERATIONS; ++it) {
        benchmark_avx2_agrp_pomc(ghrelin.data(), leptin.data(), agrp.data(), pomc.data(), mc4r.data(), ee.data(), N_CHANNELS);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> elapsed = end - start;
    double avg_us = elapsed.count() / ITERATIONS;
    double throughput = (static_cast<double>(N_CHANNELS) * ITERATIONS) / (elapsed.count() / 1e6) / 1e6;

    std::cout << "Mean Iteration Time: " << avg_us << " us" << std::endl;
    std::cout << "AVX2/FMA Throughput: " << throughput << " Million channels/sec" << std::endl;
    std::cout << "Sample AgRP: " << agrp[0] << " Hz, POMC: " << pomc[0] << " Hz, MC4R: " << mc4r[0] << "%, EE: " << ee[0] << "%" << std::endl;

    return 0;
}
