// Vectorized AVX2/FMA Microbenchmark: Target #410 Glymphatic Convective Flux & AQP4 Polarization
// Astrocytic endfoot AQP4 water transport, NREM slow-wave vasomotion, and parenchymal clearance

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>

void benchmark_glymphatic_aqp4(size_t n_elements, size_t iterations) {
    std::vector<float> pol(n_elements, 35.0f);
    std::vector<float> delta(n_elements, 0.85f);
    std::vector<float> vasomotion(n_elements, 3.8f);
    std::vector<float> shed_dose(n_elements, 100.0f);

    std::vector<float> flux_out(n_elements, 0.0f);
    std::vector<float> clearance_out(n_elements, 0.0f);
    std::vector<float> retention_out(n_elements, 0.0f);

    __m256 v_zero = _mm256_set1_ps(0.0f);
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_hundred = _mm256_set1_ps(100.0f);
    __m256 v_agrin_scale = _mm256_set1_ps(0.55f);
    __m256 v_max_pol = _mm256_set1_ps(100.0f);
    __m256 v_min_pol = _mm256_set1_ps(15.0f);
    __m256 v_alpha_base = _mm256_set1_ps(14.0f);
    __m256 v_delta_scale = _mm256_set1_ps(10.0f);
    __m256 v_dose_alpha = _mm256_set1_ps(0.20f / 100.0f);
    __m256 v_vaso_div = _mm256_set1_ps(1.0f / 3.5f);
    __m256 v_pol_div = _mm256_set1_ps(1.0f / 85.0f);
    __m256 v_alpha_div = _mm256_set1_ps(1.0f / 22.0f);
    __m256 v_darcy_k = _mm256_set1_ps(2.1f);
    __m256 v_clear_scale = _mm256_set1_ps(75.0f / 2.0f);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < n_elements; i += 8) {
            __m256 p = _mm256_loadu_ps(&pol[i]);
            __m256 d = _mm256_loadu_ps(&delta[i]);
            __m256 v = _mm256_loadu_ps(&vasomotion[i]);
            __m256 s = _mm256_loadu_ps(&shed_dose[i]);

            // Effective AQP4 polarization = min(100.0, max(15.0, p + s * 0.55))
            __m256 eff_pol = _mm256_min_ps(v_max_pol, _mm256_max_ps(v_min_pol, _mm256_fmadd_ps(s, v_agrin_scale, p)));

            // Interstitial volume fraction alpha = 14.0 + (delta * 10.0) * (1.0 + s * 0.20 / 100)
            __m256 dose_mod = _mm256_fmadd_ps(s, v_dose_alpha, v_one);
            __m256 eff_alpha = _mm256_fmadd_ps(_mm256_mul_ps(d, v_delta_scale), dose_mod, v_alpha_base);
            eff_alpha = _mm256_min_ps(_mm256_set1_ps(25.0f), _mm256_max_ps(_mm256_set1_ps(12.0f), eff_alpha));

            // Convective flux (um/min) = (v / 3.5) * (eff_pol / 85.0) * (eff_alpha / 22.0) * 2.1
            __m256 v_norm = _mm256_mul_ps(v, v_vaso_div);
            __m256 p_norm = _mm256_mul_ps(eff_pol, v_pol_div);
            __m256 a_norm = _mm256_mul_ps(eff_alpha, v_alpha_div);
            __m256 flux = _mm256_mul_ps(_mm256_mul_ps(v_norm, p_norm), _mm256_mul_ps(a_norm, v_darcy_k));
            _mm256_storeu_ps(&flux_out[i], flux);

            // Clearance (%/hr) = min(100.0, max(5.0, (flux / 2.0) * 75.0))
            __m256 clear = _mm256_min_ps(v_hundred, _mm256_max_ps(_mm256_set1_ps(5.0f), _mm256_mul_ps(flux, v_clear_scale)));
            _mm256_storeu_ps(&clearance_out[i], clear);

            // Retention = 100.0 - clear
            __m256 ret = _mm256_max_ps(v_zero, _mm256_sub_ps(v_hundred, clear));
            _mm256_storeu_ps(&retention_out[i], ret);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;

    std::cout << "Target #410 AVX2/FMA Benchmark: Glymphatic Convective Flux & AQP4 Polarization\n";
    std::cout << "Elements: " << n_elements << " | Iterations: " << iterations << "\n";
    std::cout << "Total Time: " << ms.count() << " ms\n";
    std::cout << "Throughput: " << (double)(n_elements * iterations) / (ms.count() * 1000.0) << " Melem/s\n";
    std::cout << "Sample Flux: " << flux_out[0] << " um/min | Clearance: " << clearance_out[0] << "%/hr | Retention: " << retention_out[0] << "%\n";
}

int main() {
    benchmark_glymphatic_aqp4(1048576, 10);
    return 0;
}
