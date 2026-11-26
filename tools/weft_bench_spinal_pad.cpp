#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

// WEFT AVX2 Microbenchmark: Spinal Primary Afferent Depolarization (PAD) Gating
// Benchmarks 32,768 spinal intraspinal primary afferent terminal boutons.

void benchmark_spinal_pad_avx2(size_t n, size_t iterations) {
    std::vector<float> v_desc(n);
    std::vector<float> ia_rate(n);
    std::vector<float> pad_rate_out(n, 0.0f);
    std::vector<float> terminal_depol_out(n, 0.0f);
    std::vector<float> attenuated_epsp_out(n, 0.0f);
    std::vector<float> suppression_out(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        v_desc[i] = 0.05f + 2.5f * (static_cast<float>(i) / static_cast<float>(n));
        ia_rate[i] = 20.0f + 160.0f * (static_cast<float>(i) / static_cast<float>(n));
    }

    const __m256 v_fpad_base = _mm256_set1_ps(5.0f);
    const __m256 v_kpad = _mm256_set1_ps(95.0f);
    const __m256 v_pad_max_rate = _mm256_set1_ps(250.0f);
    const __m256 v_vpad_max = _mm256_set1_ps(22.0f);
    const __m256 v_inv80 = _mm256_set1_ps(1.0f / 80.0f);
    const __m256 v_inv7_5 = _mm256_set1_ps(1.0f / 7.5f);
    const __m256 v_epsp_base = _mm256_set1_ps(5.5f);
    const __m256 v_inv100 = _mm256_set1_ps(0.01f);
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_half = _mm256_set1_ps(0.5f);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < n; i += 8) {
            __m256 v_v = _mm256_loadu_ps(&v_desc[i]);
            __m256 v_ia = _mm256_loadu_ps(&ia_rate[i]);

            // PAD rate: min(250.0, 5.0 + 95.0 * v_desc)
            __m256 raw_pad = _mm256_add_ps(v_fpad_base, _mm256_mul_ps(v_kpad, v_v));
            __m256 pad_rate = _mm256_min_ps(raw_pad, v_pad_max_rate);

            // Terminal PAD depol: 22.0 * (1.0 - 1.0 / (1.0 + x + 0.5 * x^2))
            __m256 x1 = _mm256_mul_ps(pad_rate, v_inv80);
            __m256 denom1 = _mm256_add_ps(v_one, _mm256_add_ps(x1, _mm256_mul_ps(v_half, _mm256_mul_ps(x1, x1))));
            __m256 exp1 = _mm256_div_ps(v_one, denom1);
            __m256 term_depol = _mm256_mul_ps(v_vpad_max, _mm256_sub_ps(v_one, exp1));

            // Attenuation factor: 1.0 / (1.0 + x2 + 0.5 * x2^2)
            __m256 x2 = _mm256_mul_ps(term_depol, v_inv7_5);
            __m256 denom2 = _mm256_add_ps(v_one, _mm256_add_ps(x2, _mm256_mul_ps(v_half, _mm256_mul_ps(x2, x2))));
            __m256 atten = _mm256_div_ps(v_one, denom2);

            // Attenuated EPSP: 5.5 * atten * (ia_rate / 100.0)
            __m256 epsp = _mm256_mul_ps(v_epsp_base, _mm256_mul_ps(atten, _mm256_mul_ps(v_ia, v_inv100)));
            __m256 supp = _mm256_sub_ps(v_one, atten);

            _mm256_storeu_ps(&pad_rate_out[i], pad_rate);
            _mm256_storeu_ps(&terminal_depol_out[i], term_depol);
            _mm256_storeu_ps(&attenuated_epsp_out[i], epsp);
            _mm256_storeu_ps(&suppression_out[i], supp);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;

    std::cout << "[WEFT AVX2 BENCH] Spinal Afferent Terminal Boutons: " << n << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Total Iterations: " << iterations << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Total Elapsed Time: " << elapsed.count() << " us" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Time per Iteration: " << (elapsed.count() / iterations) << " us" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Throughput: " 
              << (static_cast<double>(n * iterations) / (elapsed.count() * 1e-6)) / 1e6 << " MBoutons/sec" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Sample Target [n/2]: v=" << v_desc[n/2] << " m/s => PAD=" 
              << pad_rate_out[n/2] << " Hz, Depol=+" << terminal_depol_out[n/2] << " mV, EPSP=" 
              << attenuated_epsp_out[n/2] << " mV, Suppression=" << suppression_out[n/2] * 100.0f << "%" << std::endl;
}

int main() {
    benchmark_spinal_pad_avx2(32768, 5000);
    return 0;
}
