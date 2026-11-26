// Vectorized AVX2/FMA Microbenchmark: Target #409 Microglial Mac-1 & NLRP3 Inflammasome Gating
// Cryptic epitope gamma390-396 ligation, ASC speck assembly, and IL-1beta suppression

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>

void benchmark_microglial_mac1_inflammasome(size_t n_elements, size_t iterations) {
    std::vector<float> gamma(n_elements, 95.0f);
    std::vector<float> tlr4(n_elements, 85.0f);
    std::vector<float> p2x7(n_elements, 90.0f);
    std::vector<float> shed_dose(n_elements, 100.0f);

    std::vector<float> mac1_out(n_elements, 0.0f);
    std::vector<float> speck_out(n_elements, 0.0f);
    std::vector<float> il1b_out(n_elements, 0.0f);
    std::vector<float> ratio_out(n_elements, 0.0f);

    __m256 v_zero = _mm256_set1_ps(0.0f);
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_hundred = _mm256_set1_ps(100.0f);
    __m256 v_shield_scale = _mm256_set1_ps(0.95f / 100.0f);
    __m256 v_max_shield = _mm256_set1_ps(0.92f);
    __m256 v_silence_scale = _mm256_set1_ps(0.65f / 100.0f);
    __m256 v_mac1_scale = _mm256_set1_ps(1.05f);
    __m256 v_tlr4_w = _mm256_set1_ps(0.40f);
    __m256 v_mac1_w = _mm256_set1_ps(0.60f);
    __m256 v_prime_supp = _mm256_set1_ps(0.70f / 100.0f);
    __m256 v_act_supp = _mm256_set1_ps(0.50f / 100.0f);
    __m256 v_speck_div = _mm256_set1_ps(1.0f / 75.0f);
    __m256 v_casp_scale = _mm256_set1_ps(0.94f);
    __m256 v_il1b_base = _mm256_set1_ps(15.0f);
    __m256 v_il1b_scale = _mm256_set1_ps(5.2f);
    __m256 v_ratio_div = _mm256_set1_ps(1.0f / 15.0f);
    __m256 v_ratio_supp = _mm256_set1_ps(0.88f / 100.0f);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < n_elements; i += 8) {
            __m256 g = _mm256_loadu_ps(&gamma[i]);
            __m256 t = _mm256_loadu_ps(&tlr4[i]);
            __m256 p = _mm256_loadu_ps(&p2x7[i]);
            __m256 d = _mm256_loadu_ps(&shed_dose[i]);

            // HGF shielding = min(0.92, d * 0.95 / 100)
            __m256 shield = _mm256_min_ps(v_max_shield, _mm256_mul_ps(d, v_shield_scale));
            __m256 eff_gamma = _mm256_mul_ps(g, _mm256_sub_ps(v_one, shield));

            // Mac-1 = max(2.0, min(100.0, eff_gamma * 1.05 * (1.0 - d * 0.65 / 100)))
            __m256 itgam_mod = _mm256_sub_ps(v_one, _mm256_mul_ps(d, v_silence_scale));
            __m256 raw_mac1 = _mm256_mul_ps(_mm256_mul_ps(eff_gamma, v_mac1_scale), itgam_mod);
            __m256 mac1 = _mm256_max_ps(_mm256_set1_ps(2.0f), _mm256_min_ps(v_hundred, raw_mac1));
            _mm256_storeu_ps(&mac1_out[i], mac1);

            // Priming = (tlr4 * 0.4 + mac1 * 0.6) * (1.0 - d * 0.70 / 100)
            __m256 raw_prime = _mm256_add_ps(_mm256_mul_ps(t, v_tlr4_w), _mm256_mul_ps(mac1, v_mac1_w));
            __m256 prime = _mm256_mul_ps(raw_prime, _mm256_sub_ps(v_one, _mm256_mul_ps(d, v_prime_supp)));

            // Activation = p2x7 * (1.0 - d * 0.50 / 100)
            __m256 act = _mm256_mul_ps(p, _mm256_sub_ps(v_one, _mm256_mul_ps(d, v_act_supp)));

            // ASC speck = min(100, (prime * act) / 75.0)
            __m256 speck = _mm256_min_ps(v_hundred, _mm256_mul_ps(_mm256_mul_ps(prime, act), v_speck_div));
            _mm256_storeu_ps(&speck_out[i], speck);

            // Caspase-1 & IL-1beta
            __m256 casp1 = _mm256_mul_ps(speck, v_casp_scale);
            __m256 il1b = _mm256_min_ps(_mm256_set1_ps(600.0f), _mm256_fmadd_ps(casp1, v_il1b_scale, v_il1b_base));
            _mm256_storeu_ps(&il1b_out[i], il1b);

            // M1/M2 Ratio = max(0.12, min(6.0, (mac1 / 15.0) * (1.0 - d * 0.88 / 100)))
            __m256 raw_ratio = _mm256_mul_ps(_mm256_mul_ps(mac1, v_ratio_div), _mm256_sub_ps(v_one, _mm256_mul_ps(d, v_ratio_supp)));
            __m256 ratio = _mm256_max_ps(_mm256_set1_ps(0.12f), _mm256_min_ps(_mm256_set1_ps(6.0f), raw_ratio));
            _mm256_storeu_ps(&ratio_out[i], ratio);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;

    std::cout << "Target #409 AVX2/FMA Benchmark: Microglial Mac-1 & NLRP3 Inflammasome Gating\n";
    std::cout << "Elements: " << n_elements << " | Iterations: " << iterations << "\n";
    std::cout << "Total Time: " << ms.count() << " ms\n";
    std::cout << "Throughput: " << (double)(n_elements * iterations) / (ms.count() * 1000.0) << " Melem/s\n";
    std::cout << "Sample Mac-1: " << mac1_out[0] << "% | Specks: " << speck_out[0] << "% | IL-1b: " << il1b_out[0] << " pg/mL | M1/M2: " << ratio_out[0] << "x\n";
}

int main() {
    benchmark_microglial_mac1_inflammasome(1048576, 10);
    return 0;
}
