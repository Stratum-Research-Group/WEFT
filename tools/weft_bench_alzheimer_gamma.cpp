#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

/**
 * WEFT AVX2 / FMA Microbenchmark: Target #403
 * Alzheimer's Disease: Corticocortical Gamma Uncoupling & PV Interneuron Hypoactivity
 * Collaborative Research Target with Kevin W. McCairn, Ph.D.
 * 
 * Computes:
 * - Parvalbumin-positive (PV+) fast-spiking interneuron hypoactivity under Nav1.1 impairment
 * - Corticocortical 40 Hz narrow-band gamma power
 * - Aperiodic 1/f spectral tilt slope (chi) E/I balance
 * - Subclinical non-convulsive epileptiform micro-burst rate
 */

void compute_alzheimer_gamma_avx2(
    const float* __restrict__ path_load,
    const float* __restrict__ nav_enhancer,
    const float* __restrict__ genus_active,
    float* __restrict__ pv_rate,
    float* __restrict__ gamma_power,
    float* __restrict__ spectral_chi,
    float* __restrict__ epileptiform_spikes,
    size_t n
) {
    const __m256 v_ones = _mm256_set1_ps(1.0f);
    const __m256 v_zeros = _mm256_setzero_ps();
    const __m256 v_pv_base = _mm256_set1_ps(55.0f);
    const __m256 v_pv_scale = _mm256_set1_ps(42.0f);
    const __m256 v_nav_scale = _mm256_set1_ps(0.75f);
    const __m256 v_pv_min = _mm256_set1_ps(12.0f);
    const __m256 v_gamma_scale = _mm256_set1_ps(50.0f / 55.0f);
    const __m256 v_gamma_min = _mm256_set1_ps(6.0f);
    const __m256 v_gamma_genus_floor = _mm256_set1_ps(42.0f);
    const __m256 v_chi_base = _mm256_set1_ps(2.80f);
    const __m256 v_chi_scale = _mm256_set1_ps(1.60f);
    const __m256 v_chi_genus_boost = _mm256_set1_ps(0.60f);
    const __m256 v_chi_min = _mm256_set1_ps(1.00f);
    const __m256 v_pv_thresh = _mm256_set1_ps(30.0f);
    const __m256 v_spike_scale = _mm256_set1_ps(0.25f);
    const __m256 v_spike_genus_factor = _mm256_set1_ps(0.35f);
    const __m256 v_spike_base = _mm256_set1_ps(0.1f);
    const __m256 v_spike_max = _mm256_set1_ps(6.0f);
    const __m256 v_half = _mm256_set1_ps(0.5f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_path = _mm256_max_ps(v_zeros, _mm256_min_ps(v_ones, _mm256_loadu_ps(&path_load[i])));
        __m256 v_nav = _mm256_max_ps(v_zeros, _mm256_min_ps(v_ones, _mm256_loadu_ps(&nav_enhancer[i])));
        __m256 v_genus = _mm256_loadu_ps(&genus_active[i]);
        __m256 genus_mask = _mm256_cmp_ps(v_genus, v_half, _CMP_GT_OQ);

        // Effective nav loss = path_load * (1 - 0.75 * nav_enhancer)
        __m256 nav_loss = _mm256_mul_ps(v_path, _mm256_sub_ps(v_ones, _mm256_mul_ps(v_nav, v_nav_scale)));

        // PV firing rate = max(12.0, 55.0 - 42.0 * nav_loss)
        __m256 pv = _mm256_max_ps(v_pv_min, _mm256_fnmadd_ps(v_pv_scale, nav_loss, v_pv_base));
        _mm256_storeu_ps(&pv_rate[i], pv);

        // Gamma power = max(6.0, (pv / 55.0) * 50.0), clamped by genus to 42.0
        __m256 raw_gamma = _mm256_mul_ps(pv, v_gamma_scale);
        __m256 genus_clamped_gamma = _mm256_blendv_ps(raw_gamma, _mm256_max_ps(raw_gamma, v_gamma_genus_floor), genus_mask);
        __m256 gamma = _mm256_max_ps(v_gamma_min, genus_clamped_gamma);
        _mm256_storeu_ps(&gamma_power[i], gamma);

        // Spectral slope chi = 2.80 - 1.60 * nav_loss + (if genus then 0.60 else 0)
        __m256 base_chi = _mm256_fnmadd_ps(v_chi_scale, nav_loss, v_chi_base);
        __m256 chi_boost = _mm256_blendv_ps(v_zeros, v_chi_genus_boost, genus_mask);
        __m256 chi = _mm256_max_ps(v_chi_min, _mm256_add_ps(base_chi, chi_boost));
        _mm256_storeu_ps(&spectral_chi[i], chi);

        // Epileptiform micro-spikes
        __m256 pv_deficit = _mm256_max_ps(v_zeros, _mm256_sub_ps(v_pv_thresh, pv));
        __m256 spike_atten = _mm256_blendv_ps(v_ones, v_spike_genus_factor, genus_mask);
        __m256 raw_spikes = _mm256_min_ps(v_spike_max, _mm256_mul_ps(v_spike_scale, _mm256_mul_ps(pv_deficit, spike_atten)));
        __m256 spikes_cond = _mm256_cmp_ps(pv, v_pv_thresh, _CMP_LT_OQ);
        __m256 spikes = _mm256_blendv_ps(v_spike_base, raw_spikes, spikes_cond);
        _mm256_storeu_ps(&epileptiform_spikes[i], spikes);
    }
}

int main() {
    const size_t N = 1000000;
    std::vector<float> path(N), nav(N), genus(N);
    std::vector<float> pv(N), gamma(N), chi(N), spikes(N);

    // Populate disease trajectory
    for (size_t i = 0; i < N; ++i) {
        float frac = static_cast<float>(i) / static_cast<float>(N);
        path[i] = (frac < 0.6f) ? frac / 0.6f * 0.85f : 0.85f;
        nav[i] = (frac > 0.8f) ? (frac - 0.8f) / 0.2f * 0.85f : 0.0f;
        genus[i] = (frac > 0.8f) ? 1.0f : 0.0f;
    }

    // Warmup
    compute_alzheimer_gamma_avx2(
        path.data(), nav.data(), genus.data(),
        pv.data(), gamma.data(), chi.data(), spikes.data(), N
    );

    const int WARMUP_RUNS = 20;
    const int BENCH_RUNS = 100;
    for (int r = 0; r < WARMUP_RUNS; ++r) {
        compute_alzheimer_gamma_avx2(
            path.data(), nav.data(), genus.data(),
            pv.data(), gamma.data(), chi.data(), spikes.data(), N
        );
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < BENCH_RUNS; ++r) {
        compute_alzheimer_gamma_avx2(
            path.data(), nav.data(), genus.data(),
            pv.data(), gamma.data(), chi.data(), spikes.data(), N
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / BENCH_RUNS;
    double throughput_melem = (static_cast<double>(N) / (avg_ms / 1000.0)) / 1e6;

    std::cout << "=================================================================\n";
    std::cout << "WEFT AVX2/FMA Benchmark: Target #403 Alzheimer Gamma Uncoupling\n";
    std::cout << "=================================================================\n";
    std::cout << "Vector Size:           " << N << " elements\n";
    std::cout << "Average Latency:       " << avg_ms << " ms per million state evaluations\n";
    std::cout << "Throughput:            " << throughput_melem << " Melem/s\n";
    std::cout << "Sample Trace:\n";
    std::cout << "  [Healthy Baseline]  Path: " << path[0]*100.0f << "% | PV+: " << pv[0] << " Hz | Gamma: " << gamma[0] << " uV^2 | Chi: " << chi[0] << " | Spikes: " << spikes[0] << " Hz\n";
    std::cout << "  [Severe AD Loss]    Path: " << path[700000]*100.0f << "% | PV+: " << pv[700000] << " Hz | Gamma: " << gamma[700000] << " uV^2 | Chi: " << chi[700000] << " | Spikes: " << spikes[700000] << " Hz\n";
    std::cout << "  [GENUS & Nav1.1]    GENUS: " << genus[950000] << " | PV+: " << pv[950000] << " Hz | Gamma: " << gamma[950000] << " uV^2 | Chi: " << chi[950000] << " | Spikes: " << spikes[950000] << " Hz\n";
    std::cout << "=================================================================\n";

    return 0;
}
