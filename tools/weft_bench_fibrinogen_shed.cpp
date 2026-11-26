// Vectorized AVX2/FMA Microbenchmark: Target #406 Fibrinogen-SHED-CM Molecular Interactions
// Anti-amyloidogenic fibrin clot shielding and microglial Mac-1 suppression

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>

void benchmark_fibrinogen_shed(size_t n_elements, size_t iterations) {
    std::vector<float> fib(n_elements, 3.2f);
    std::vector<float> thr(n_elements, 1.85f);
    std::vector<float> abeta(n_elements, 85.0f);
    std::vector<float> shed_dose(n_elements, 100.0f);

    std::vector<float> clot_out(n_elements, 0.0f);
    std::vector<float> mac1_out(n_elements, 0.0f);
    std::vector<float> spine_out(n_elements, 0.0f);

    __m256 v_zero = _mm256_set1_ps(0.0f);
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_hundred = _mm256_set1_ps(100.0f);
    __m256 v_fib_div = _mm256_set1_ps(1.0f / 3.0f);
    __m256 v_thr_div = _mm256_set1_ps(1.0f / 1.5f);
    __m256 v_abeta_scale = _mm256_set1_ps(1.0f / 50.0f);
    __m256 v_base_scale = _mm256_set1_ps(35.0f);
    __m256 v_hgf_weight = _mm256_set1_ps(0.55f / 100.0f);
    __m256 v_clust_weight = _mm256_set1_ps(0.40f / 100.0f);
    __m256 v_max_shield = _mm256_set1_ps(0.92f);
    __m256 v_mac1_scale = _mm256_set1_ps(0.92f);
    __m256 v_supp_scale = _mm256_set1_ps(0.35f / 100.0f);
    __m256 v_spine_scale = _mm256_set1_ps(0.82f);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < n_elements; i += 8) {
            __m256 f = _mm256_loadu_ps(&fib[i]);
            __m256 t = _mm256_loadu_ps(&thr[i]);
            __m256 a = _mm256_loadu_ps(&abeta[i]);
            __m256 d = _mm256_loadu_ps(&shed_dose[i]);

            // Raw clot drive = (f/3.0) * (t/1.5) * (1.0 + a/50.0) * 35.0
            __m256 f_part = _mm256_mul_ps(f, v_fib_div);
            __m256 t_part = _mm256_mul_ps(t, v_thr_div);
            __m256 a_part = _mm256_fmadd_ps(a, v_abeta_scale, v_one);
            __m256 raw_clot = _mm256_mul_ps(_mm256_mul_ps(f_part, t_part), _mm256_mul_ps(a_part, v_base_scale));
            raw_clot = _mm256_min_ps(v_hundred, raw_clot);

            // Shielding = min(0.92, dose*(0.55/100 + 0.40/100))
            __m256 shield = _mm256_min_ps(v_max_shield, _mm256_add_ps(_mm256_mul_ps(d, v_hgf_weight), _mm256_mul_ps(d, v_clust_weight)));

            // Effective clot = raw_clot * (1.0 - shield)
            __m256 eff_clot = _mm256_mul_ps(raw_clot, _mm256_sub_ps(v_one, shield));
            _mm256_storeu_ps(&clot_out[i], eff_clot);

            // Mac-1 = eff_clot * 0.92 * (1.0 - dose * 0.35/100)
            __m256 mac1_supp = _mm256_sub_ps(v_one, _mm256_mul_ps(d, v_supp_scale));
            __m256 mac1 = _mm256_mul_ps(_mm256_mul_ps(eff_clot, v_mac1_scale), mac1_supp);
            _mm256_storeu_ps(&mac1_out[i], mac1);

            // Spine loss = mac1 * 0.82
            __m256 spine_loss = _mm256_mul_ps(mac1, v_spine_scale);
            _mm256_storeu_ps(&spine_out[i], spine_loss);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;

    std::cout << "Target #406 AVX2/FMA Benchmark: Fibrinogen & SHED-CM Molecular Interactions\n";
    std::cout << "Elements: " << n_elements << " | Iterations: " << iterations << "\n";
    std::cout << "Total Time: " << ms.count() << " ms\n";
    std::cout << "Throughput: " << (double)(n_elements * iterations) / (ms.count() * 1000.0) << " Melem/s\n";
    std::cout << "Sample Clot: " << clot_out[0] << "% | Mac-1: " << mac1_out[0] << "% | Spine Loss: " << spine_out[0] << "%\n";
}

int main() {
    benchmark_fibrinogen_shed(1048576, 10);
    return 0;
}
