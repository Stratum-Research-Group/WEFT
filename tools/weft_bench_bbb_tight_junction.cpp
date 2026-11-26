// Vectorized AVX2/FMA Microbenchmark: Target #407 Blood-Brain Barrier Tight-Junction Restoration
// Endothelial Tie2 signaling, Claudin-5/Occludin reconstitution, and TEER electrical resistance

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>

void benchmark_bbb_tight_junction(size_t n_elements, size_t iterations) {
    std::vector<float> pericyte(n_elements, 85.0f);
    std::vector<float> ang1(n_elements, 18.0f);
    std::vector<float> ros(n_elements, 20.0f);
    std::vector<float> mmp(n_elements, 1.5f);
    std::vector<float> shed_dose(n_elements, 100.0f);

    std::vector<float> tie2_out(n_elements, 0.0f);
    std::vector<float> claudin_out(n_elements, 0.0f);
    std::vector<float> teer_out(n_elements, 0.0f);
    std::vector<float> leak_out(n_elements, 0.0f);

    __m256 v_zero = _mm256_set1_ps(0.0f);
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_hundred = _mm256_set1_ps(100.0f);
    __m256 v_base_tie2 = _mm256_set1_ps(20.0f);
    __m256 v_peri_scale = _mm256_set1_ps(0.3f);
    __m256 v_ang_scale = _mm256_set1_ps(3.2f);
    __m256 v_ros_scale = _mm256_set1_ps(1.0f / 200.0f);
    __m256 v_supp_scale = _mm256_set1_ps(0.75f / 100.0f);
    __m256 v_min_supp = _mm256_set1_ps(0.2f);
    __m256 v_claud_t_scale = _mm256_set1_ps(1.1f);
    __m256 v_claud_m_scale = _mm256_set1_ps(12.0f);
    __m256 v_occl_t_scale = _mm256_set1_ps(1.05f);
    __m256 v_occl_m_scale = _mm256_set1_ps(10.5f);
    __m256 v_teer_base = _mm256_set1_ps(200.0f);
    __m256 v_claud_teer = _mm256_set1_ps(8.5f);
    __m256 v_occl_teer = _mm256_set1_ps(4.5f);
    __m256 v_teer_div = _mm256_set1_ps(100.0f / 1500.0f);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < n_elements; i += 8) {
            __m256 p = _mm256_loadu_ps(&pericyte[i]);
            __m256 a = _mm256_loadu_ps(&ang1[i]);
            __m256 r = _mm256_loadu_ps(&ros[i]);
            __m256 m = _mm256_loadu_ps(&mmp[i]);
            __m256 d = _mm256_loadu_ps(&shed_dose[i]);

            // Tie2 = min(100, max(10, (20.0 + p*0.3 + a*3.2) * (1.0 - r/200.0)))
            __m256 raw_t = _mm256_add_ps(v_base_tie2, _mm256_fmadd_ps(p, v_peri_scale, _mm256_mul_ps(a, v_ang_scale)));
            __m256 ros_factor = _mm256_sub_ps(v_one, _mm256_mul_ps(r, v_ros_scale));
            __m256 eff_t = _mm256_min_ps(v_hundred, _mm256_max_ps(_mm256_set1_ps(10.0f), _mm256_mul_ps(raw_t, ros_factor)));
            _mm256_storeu_ps(&tie2_out[i], eff_t);

            // MMP suppression = max(0.2, 1.0 - d * 0.75/100)
            __m256 m_supp = _mm256_max_ps(v_min_supp, _mm256_sub_ps(v_one, _mm256_mul_ps(d, v_supp_scale)));
            __m256 eff_m = _mm256_mul_ps(m, m_supp);

            // Claudin-5 = min(100, max(5, eff_t * 1.1 - eff_m * 12.0))
            __m256 claudin = _mm256_min_ps(v_hundred, _mm256_max_ps(_mm256_set1_ps(5.0f), _mm256_fmsub_ps(eff_t, v_claud_t_scale, _mm256_mul_ps(eff_m, v_claud_m_scale))));
            _mm256_storeu_ps(&claudin_out[i], claudin);

            // Occludin = min(100, max(5, eff_t * 1.05 - eff_m * 10.5))
            __m256 occludin = _mm256_min_ps(v_hundred, _mm256_max_ps(_mm256_set1_ps(5.0f), _mm256_fmsub_ps(eff_t, v_occl_t_scale, _mm256_mul_ps(eff_m, v_occl_m_scale))));

            // TEER = 200.0 + claudin*8.5 + occludin*4.5
            __m256 teer = _mm256_add_ps(v_teer_base, _mm256_fmadd_ps(claudin, v_claud_teer, _mm256_mul_ps(occludin, v_occl_teer)));
            _mm256_storeu_ps(&teer_out[i], teer);

            // Leakage = max(2.0, min(100.0, 100.0 - teer * (100.0 / 1500.0)))
            __m256 leak = _mm256_max_ps(_mm256_set1_ps(2.0f), _mm256_min_ps(v_hundred, _mm256_sub_ps(v_hundred, _mm256_mul_ps(teer, v_teer_div))));
            _mm256_storeu_ps(&leak_out[i], leak);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;

    std::cout << "Target #407 AVX2/FMA Benchmark: BBB Tight-Junction Restoration\n";
    std::cout << "Elements: " << n_elements << " | Iterations: " << iterations << "\n";
    std::cout << "Total Time: " << ms.count() << " ms\n";
    std::cout << "Throughput: " << (double)(n_elements * iterations) / (ms.count() * 1000.0) << " Melem/s\n";
    std::cout << "Sample Tie2: " << tie2_out[0] << "% | Claudin-5: " << claudin_out[0] << "% | TEER: " << teer_out[0] << " Ohm*cm^2 | Leakage: " << leak_out[0] << "%\n";
}

int main() {
    benchmark_bbb_tight_junction(1048576, 10);
    return 0;
}
