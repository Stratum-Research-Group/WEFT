#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

// WEFT AVX2 Microbenchmark: Dentato-Thalamo-Cortical (DTC) Synchronization
// Benchmarks 32,768 dentato-thalamocortical relay and M1 Layer V pyramidal units.

void benchmark_dtc_sync_avx2(size_t n, size_t iterations) {
    std::vector<float> dentate_hz(n);
    std::vector<float> cortex_phase(n);
    std::vector<float> vl_epsp_out(n, 0.0f);
    std::vector<float> m1_facil_out(n, 0.0f);
    std::vector<float> coherence_out(n, 0.0f);
    std::vector<float> sync_mode_out(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        dentate_hz[i] = 40.0f + 180.0f * (static_cast<float>(i) / static_cast<float>(n));
        cortex_phase[i] = -3.14159f + 6.28318f * (static_cast<float>(i) / static_cast<float>(n));
    }

    const __m256 v_vmax = _mm256_set1_ps(25.0f);
    const __m256 v_inv90 = _mm256_set1_ps(1.0f / 90.0f);
    const __m256 v_inv120 = _mm256_set1_ps(1.0f / 120.0f);
    const __m256 v_ktc = _mm256_set1_ps(0.85f);
    const __m256 v_half = _mm256_set1_ps(0.5f);
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_rate_th = _mm256_set1_ps(80.0f);
    const __m256 v_phase_th = _mm256_set1_ps(0.80f);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < n; i += 8) {
            __m256 vd = _mm256_loadu_ps(&dentate_hz[i]);
            __m256 vp = _mm256_loadu_ps(&cortex_phase[i]);

            // Thalamic EPSP: 25.0 * (1.0 - 1.0 / (1.0 + x + 0.5 * x^2))
            __m256 x1 = _mm256_mul_ps(vd, v_inv90);
            __m256 denom1 = _mm256_add_ps(v_one, _mm256_add_ps(x1, _mm256_mul_ps(v_half, _mm256_mul_ps(x1, x1))));
            __m256 exp1 = _mm256_div_ps(v_one, denom1);
            __m256 vl_epsp = _mm256_mul_ps(v_vmax, _mm256_sub_ps(v_one, exp1));

            // Cosine phase modulation polynomial: cos(p) ~ 1.0 - 0.5 * p^2 + 0.0416 * p^4
            __m256 p2 = _mm256_mul_ps(vp, vp);
            __m256 cos_p = _mm256_add_ps(v_one, _mm256_mul_ps(p2, _mm256_add_ps(_mm256_set1_ps(-0.5f), _mm256_mul_ps(p2, _mm256_set1_ps(0.04166f)))));
            __m256 cos_p_clamped = _mm256_max_ps(_mm256_set1_ps(-1.0f), _mm256_min_ps(v_one, cos_p));
            __m256 phase_fac = _mm256_mul_ps(v_half, _mm256_add_ps(v_one, cos_p_clamped));

            // M1 Layer V facilitation: vl_epsp * 0.85 * phase_fac
            __m256 m1_facil = _mm256_mul_ps(vl_epsp, _mm256_mul_ps(v_ktc, phase_fac));

            // Coherence & phase locking
            __m256 ge_rate = _mm256_cmp_ps(vd, v_rate_th, _CMP_GE_OQ);
            __m256 ge_phase = _mm256_cmp_ps(phase_fac, v_phase_th, _CMP_GE_OQ);
            __m256 is_locked_mask = _mm256_and_ps(ge_rate, ge_phase);
            __m256 is_locked = _mm256_blendv_ps(v_zero, v_one, is_locked_mask);

            _mm256_storeu_ps(&vl_epsp_out[i], vl_epsp);
            _mm256_storeu_ps(&m1_facil_out[i], m1_facil);
            _mm256_storeu_ps(&coherence_out[i], phase_fac);
            _mm256_storeu_ps(&sync_mode_out[i], is_locked);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;

    std::cout << "[WEFT AVX2 BENCH] Dentato-Thalamocortical Units: " << n << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Total Iterations: " << iterations << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Total Elapsed Time: " << elapsed.count() << " us" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Time per Iteration: " << (elapsed.count() / iterations) << " us" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Throughput: " 
              << (static_cast<double>(n * iterations) / (elapsed.count() * 1e-6)) / 1e6 << " MUnits/sec" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Sample Target [n/2]: Dentate=" << dentate_hz[n/2] << " Hz, Phase=" 
              << cortex_phase[n/2] << " rad => VL EPSP=" << vl_epsp_out[n/2] << " mV, M1 Facil=" 
              << m1_facil_out[n/2] << " mV, Sync=" << (sync_mode_out[n/2] > 0.5f ? "LOCKED" : "SLIP") << std::endl;
}

int main() {
    benchmark_dtc_sync_avx2(32768, 5000);
    return 0;
}
