/**
 * tools/weft_bench_vlpag_survival.cpp
 * AVX2/FMA Native Vectorized SIMD Microbenchmark
 * Ventrolateral Periaqueductal Gray (vlPAG) Freezing vs Escape Arbitration
 * Lean 4 Formal Target #395 | Paper #204 | Cycle v340.73.0
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>
#include <iomanip>

void run_pag_survival_avx2_fma(
    const float* __restrict__ threat_dist,
    const float* __restrict__ looming_speed,
    const float* __restrict__ corticosterone,
    float* __restrict__ out_vl_rate,
    float* __restrict__ out_dl_rate,
    float* __restrict__ out_sympathetic,
    float* __restrict__ out_freeze_prob,
    float* __restrict__ out_escape_prob,
    size_t count
) {
    const __m256 c_one         = _mm256_set1_ps(1.0f);
    const __m256 c_zero        = _mm256_setzero_ps();
    const __m256 c_vl_base     = _mm256_set1_ps(3.0f);
    const __m256 c_vl_dist     = _mm256_set1_ps(35.0f);
    const __m256 c_vl_cort     = _mm256_set1_ps(15.0f);
    const __m256 c_vl_max      = _mm256_set1_ps(50.0f);

    const __m256 c_dl_base     = _mm256_set1_ps(2.0f);
    const __m256 c_dl_prox     = _mm256_set1_ps(60.0f);
    const __m256 c_dl_loom     = _mm256_set1_ps(40.0f);
    const __m256 c_dl_max      = _mm256_set1_ps(80.0f);

    const __m256 c_inh_dl_on_vl = _mm256_set1_ps(0.45f);
    const __m256 c_inh_vl_on_dl = _mm256_set1_ps(0.20f);

    const __m256 c_sym_dl      = _mm256_set1_ps(0.12f / 10.0f);
    const __m256 c_sym_vl      = _mm256_set1_ps(0.02f / 10.0f);

    const __m256 c_half        = _mm256_set1_ps(0.5f);
    const __m256 c_freeze_th   = _mm256_set1_ps(20.0f);
    const __m256 c_freeze_slope= _mm256_set1_ps(0.10f);
    const __m256 c_escape_th   = _mm256_set1_ps(35.0f);
    const __m256 c_escape_slope= _mm256_set1_ps(0.08f);

    for (size_t i = 0; i < count; i += 8) {
        __m256 v_dist = _mm256_loadu_ps(threat_dist + i);
        __m256 v_loom = _mm256_loadu_ps(looming_speed + i);
        __m256 v_cort = _mm256_loadu_ps(corticosterone + i);

        // 1. Raw vlPAG drive: 3.0 + 35.0 * dist * (1.0 - loom) + 15.0 * cort
        __m256 v_non_loom = _mm256_sub_ps(c_one, v_loom);
        __m256 v_dist_loom = _mm256_mul_ps(v_dist, v_non_loom);
        __m256 v_raw_vl = _mm256_fmadd_ps(c_vl_dist, v_dist_loom, c_vl_base);
        v_raw_vl = _mm256_fmadd_ps(c_vl_cort, v_cort, v_raw_vl);

        // 2. Raw dlPAG drive: 2.0 + 60.0 * (1.0 - dist) + 40.0 * loom
        __m256 v_prox = _mm256_sub_ps(c_one, v_dist);
        __m256 v_raw_dl = _mm256_fmadd_ps(c_dl_prox, v_prox, c_dl_base);
        v_raw_dl = _mm256_fmadd_ps(c_dl_loom, v_loom, v_raw_dl);

        // 3. Mutual cross-inhibition
        __m256 v_eff_vl = _mm256_fnmadd_ps(c_inh_dl_on_vl, v_raw_dl, v_raw_vl);
        v_eff_vl = _mm256_max_ps(c_zero, _mm256_min_ps(c_vl_max, v_eff_vl));

        __m256 v_eff_dl = _mm256_fnmadd_ps(c_inh_vl_on_dl, v_raw_vl, v_raw_dl);
        v_eff_dl = _mm256_max_ps(c_zero, _mm256_min_ps(c_dl_max, v_eff_dl));

        // 4. Autonomic sympathetic tone: clamp(sym_dl * dl + sym_vl * vl, 0.0, 1.0)
        __m256 v_sym = _mm256_mul_ps(v_eff_dl, c_sym_dl);
        v_sym = _mm256_fmadd_ps(v_eff_vl, c_sym_vl, v_sym);
        v_sym = _mm256_max_ps(c_zero, _mm256_min_ps(c_one, v_sym));

        // 5. Attractor probabilities
        __m256 v_freeze = _mm256_fmadd_ps(c_freeze_slope, _mm256_sub_ps(v_eff_vl, c_freeze_th), c_half);
        v_freeze = _mm256_max_ps(c_zero, _mm256_min_ps(c_one, v_freeze));

        __m256 v_escape = _mm256_fmadd_ps(c_escape_slope, _mm256_sub_ps(v_eff_dl, c_escape_th), c_half);
        v_escape = _mm256_max_ps(c_zero, _mm256_min_ps(c_one, v_escape));

        _mm256_storeu_ps(out_vl_rate + i, v_eff_vl);
        _mm256_storeu_ps(out_dl_rate + i, v_eff_dl);
        _mm256_storeu_ps(out_sympathetic + i, v_sym);
        _mm256_storeu_ps(out_freeze_prob + i, v_freeze);
        _mm256_storeu_ps(out_escape_prob + i, v_escape);
    }
}

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "WEFT AVX2/FMA SIMD Microbenchmark: vlPAG vs dlPAG Defense Gate" << std::endl;
    std::cout << "Lean 4 Target #395 | Paper #204 | Cycle v340.73.0" << std::endl;
    std::cout << "================================================================================" << std::endl;

    const size_t N = 1000000;
    std::vector<float> dist(N, 0.20f);
    std::vector<float> loom(N, 0.85f);
    std::vector<float> cort(N, 0.40f);

    std::vector<float> out_vl(N);
    std::vector<float> out_dl(N);
    std::vector<float> out_sym(N);
    std::vector<float> out_freeze(N);
    std::vector<float> out_escape(N);

    // Warmup
    for (int w = 0; w < 5; ++w) {
        run_pag_survival_avx2_fma(dist.data(), loom.data(), cort.data(),
                                  out_vl.data(), out_dl.data(), out_sym.data(),
                                  out_freeze.data(), out_escape.data(), N);
    }

    // Benchmark
    const int RUNS = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        run_pag_survival_avx2_fma(dist.data(), loom.data(), cort.data(),
                                  out_vl.data(), out_dl.data(), out_sym.data(),
                                  out_freeze.data(), out_escape.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / RUNS;
    double throughput = (double)N / (avg_ms * 1e-3) / 1e6;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Processed Elements:  " << N << " (8-lane AVX2/FMA SIMD)" << std::endl;
    std::cout << "Average Latency:     " << avg_ms << " ms" << std::endl;
    std::cout << "Peak Throughput:     " << throughput << " Melem/s (" << (throughput / 1e3) << " Gelem/s)" << std::endl;
    std::cout << "\nSample Verification at index 0:" << std::endl;
    std::cout << "  Threat Distance:     " << dist[0] * 100.0f << "%" << std::endl;
    std::cout << "  Looming Velocity:    " << loom[0] * 100.0f << "%" << std::endl;
    std::cout << "  Corticosterone:      " << cort[0] * 100.0f << "%" << std::endl;
    std::cout << "  vlPAG Rate (Freeze): " << out_vl[0] << " Hz" << std::endl;
    std::cout << "  dlPAG Rate (Escape): " << out_dl[0] << " Hz" << std::endl;
    std::cout << "  Sympathetic Tone:    " << out_sym[0] * 100.0f << "%" << std::endl;
    std::cout << "  Freeze Probability:  " << out_freeze[0] * 100.0f << "%" << std::endl;
    std::cout << "  Escape Probability:  " << out_escape[0] * 100.0f << "%" << std::endl;
    std::cout << "================================================================================" << std::endl;

    return 0;
}
