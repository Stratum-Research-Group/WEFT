// Vectorized AVX2/FMA Microbenchmark: Target #404 Paroxysmal Dyskinesia & Channelopathy
// Cerebellar-thalamo-striatal disynaptic crosstalk & carbamazepine rescue dynamics

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>

void benchmark_paroxysmal_dyskinesia(size_t n_elements, size_t iterations) {
    std::vector<float> prrt2(n_elements, 0.20f);
    std::vector<float> trigger(n_elements, 0.90f);
    std::vector<float> carbamazepine(n_elements, 300.0f);
    std::vector<float> burst_rate(n_elements, 0.0f);
    std::vector<float> da_efflux(n_elements, 0.0f);
    std::vector<float> severity(n_elements, 0.0f);

    __m256 v_dcn_base = _mm256_set1_ps(40.0f);
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_scale = _mm256_set1_ps(120.0f);
    __m256 v_cbz_div = _mm256_set1_ps(1.0f / 300.0f);
    __m256 v_cbz_factor = _mm256_set1_ps(0.75f);
    __m256 v_da_scale = _mm256_set1_ps(1.2f / 60.0f);
    __m256 v_zero = _mm256_set1_ps(0.0f);
    __m256 v_hundred = _mm256_set1_ps(100.0f);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < n_elements; i += 8) {
            __m256 p = _mm256_loadu_ps(&prrt2[i]);
            __m256 tr = _mm256_loadu_ps(&trigger[i]);
            __m256 cbz = _mm256_loadu_ps(&carbamazepine[i]);

            // Deficit = max(0, 1.0 - PRRT2)
            __m256 def = _mm256_max_ps(v_zero, _mm256_sub_ps(v_one, p));

            // Raw burst = 40.0 + def * 120.0 * trigger
            __m256 raw_burst = _mm256_fmadd_ps(_mm256_mul_ps(def, tr), v_scale, v_dcn_base);

            // Carbamazepine Nav inhibition = min(1.0, cbz / 300.0)
            __m256 nav_inh = _mm256_min_ps(v_one, _mm256_mul_ps(cbz, v_cbz_div));

            // Rescued burst = raw_burst * (1.0 - 0.75 * nav_inh)
            __m256 mod_factor = _mm256_sub_ps(v_one, _mm256_mul_ps(v_cbz_factor, nav_inh));
            __m256 b_rate = _mm256_mul_ps(raw_burst, mod_factor);
            _mm256_storeu_ps(&burst_rate[i], b_rate);

            // Striatal DA = 1.0 + (b_rate / 60.0) * 1.2
            __m256 da = _mm256_fmadd_ps(b_rate, v_da_scale, v_one);
            _mm256_storeu_ps(&da_efflux[i], da);

            // Dyskinesia severity = clamp(0, 100, (b_rate - 45)*0.95 + (da - 1.2)*22.0)
            __m256 b_excess = _mm256_mul_ps(_mm256_sub_ps(b_rate, _mm256_set1_ps(45.0f)), _mm256_set1_ps(0.95f));
            __m256 da_excess = _mm256_mul_ps(_mm256_sub_ps(da, _mm256_set1_ps(1.2f)), _mm256_set1_ps(22.0f));
            __m256 raw_sev = _mm256_add_ps(b_excess, da_excess);
            __m256 sev = _mm256_max_ps(v_zero, _mm256_min_ps(v_hundred, raw_sev));
            _mm256_storeu_ps(&severity[i], sev);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;

    std::cout << "Target #404 AVX2/FMA Benchmark: Paroxysmal Dyskinesia & Channelopathy\n";
    std::cout << "Elements: " << n_elements << " | Iterations: " << iterations << "\n";
    std::cout << "Total Time: " << ms.count() << " ms\n";
    std::cout << "Throughput: " << (double)(n_elements * iterations) / (ms.count() * 1000.0) << " Melem/s\n";
    std::cout << "Sample Burst: " << burst_rate[0] << " Hz | DA: " << da_efflux[0] << "x | Severity: " << severity[0] << "/100\n";
}

int main() {
    benchmark_paroxysmal_dyskinesia(1048576, 10);
    return 0;
}
