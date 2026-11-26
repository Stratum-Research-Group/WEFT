#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

/**
 * WEFT AVX2 / FMA Microbenchmark: Target #402
 * Essential Tremor (ET): Inferior Olivary Pacemaking & DRT Hyper-Oscillation
 * Collaborative Research Target with Kevin W. McCairn, Ph.D.
 * 
 * Computes:
 * - Inferior olivary low-threshold T-type Ca2+ pacemaking frequency (4 - 8 Hz)
 * - Connexin-36 electrotonic gap-junctional coupling
 * - Dentato-Rubro-Thalamic (DRT) tract oscillatory coherence
 * - VIM thalamic relay burst drive & peripheral hand tremor amplitude (cm)
 */

void compute_essential_tremor_avx2(
    const float* __restrict__ gap_coupling,
    const float* __restrict__ cav3_cond,
    const float* __restrict__ cav3_block,
    const float* __restrict__ vim_dbs,
    float* __restrict__ ion_freq,
    float* __restrict__ drt_coherence,
    float* __restrict__ thal_power,
    float* __restrict__ tremor_amp,
    size_t n
) {
    const __m256 v_ones = _mm256_set1_ps(1.0f);
    const __m256 v_zeros = _mm256_setzero_ps();
    const __m256 v_cav3_scale = _mm256_set1_ps(0.85f);
    const __m256 v_gap_scale = _mm256_set1_ps(0.60f);
    const __m256 v_freq_base = _mm256_set1_ps(4.2f);
    const __m256 v_freq_scale = _mm256_set1_ps(2.5f);
    const __m256 v_cav3_thresh = _mm256_set1_ps(0.20f);
    const __m256 v_coh_high = _mm256_set1_ps(0.90f);
    const __m256 v_coh_low = _mm256_set1_ps(0.20f);
    const __m256 v_dbs_factor = _mm256_set1_ps(0.15f);
    const __m256 v_pow_cav3 = _mm256_set1_ps(40.0f);
    const __m256 v_pow_gap = _mm256_set1_ps(45.0f);
    const __m256 v_pow_min = _mm256_set1_ps(5.0f);
    const __m256 v_amp_scale = _mm256_set1_ps(3.8f);
    const __m256 v_amp_min = _mm256_set1_ps(0.1f);
    const __m256 v_half = _mm256_set1_ps(0.5f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_gap = _mm256_max_ps(v_zeros, _mm256_min_ps(v_ones, _mm256_loadu_ps(&gap_coupling[i])));
        __m256 v_cav = _mm256_max_ps(v_zeros, _mm256_min_ps(v_ones, _mm256_loadu_ps(&cav3_cond[i])));
        __m256 v_blk = _mm256_max_ps(v_zeros, _mm256_min_ps(v_ones, _mm256_loadu_ps(&cav3_block[i])));
        __m256 v_dbs = _mm256_loadu_ps(&vim_dbs[i]);

        // Effective Cav3 and gap coupling
        __m256 eff_cav = _mm256_mul_ps(v_cav, _mm256_sub_ps(v_ones, _mm256_mul_ps(v_blk, v_cav3_scale)));
        __m256 eff_gap = _mm256_mul_ps(v_gap, _mm256_sub_ps(v_ones, _mm256_mul_ps(v_blk, v_gap_scale)));

        // ION frequency = 4.2 + 2.5 * eff_cav
        __m256 freq = _mm256_fmadd_ps(eff_cav, v_freq_scale, v_freq_base);
        _mm256_storeu_ps(&ion_freq[i], freq);

        // DRT coherence
        __m256 cav_mask = _mm256_cmp_ps(eff_cav, v_cav3_thresh, _CMP_GT_OQ);
        __m256 coh_mult = _mm256_blendv_ps(v_coh_low, v_coh_high, cav_mask);
        __m256 raw_coh = _mm256_mul_ps(eff_gap, coh_mult);

        // VIM DBS modulation
        __m256 dbs_active_mask = _mm256_cmp_ps(v_dbs, v_half, _CMP_GT_OQ);
        __m256 dbs_atten = _mm256_blendv_ps(v_ones, v_dbs_factor, dbs_active_mask);
        __m256 final_coh = _mm256_min_ps(v_ones, _mm256_mul_ps(raw_coh, dbs_atten));
        _mm256_storeu_ps(&drt_coherence[i], final_coh);

        // Thalamic tremor power = max(5.0, (eff_cav * 40 + eff_gap * 45) * dbs_atten)
        __m256 raw_pow = _mm256_add_ps(_mm256_mul_ps(eff_cav, v_pow_cav3), _mm256_mul_ps(eff_gap, v_pow_gap));
        __m256 pow = _mm256_max_ps(v_pow_min, _mm256_mul_ps(raw_pow, dbs_atten));
        _mm256_storeu_ps(&thal_power[i], pow);

        // Tremor amp = max(0.1, final_coh * 3.8)
        __m256 amp = _mm256_max_ps(v_amp_min, _mm256_mul_ps(final_coh, v_amp_scale));
        _mm256_storeu_ps(&tremor_amp[i], amp);
    }
}

int main() {
    const size_t N = 1000000;
    std::vector<float> gap(N), cav(N), blk(N), dbs(N);
    std::vector<float> freq(N), coh(N), pow(N), amp(N);

    // Populate disease and treatment regimes
    for (size_t i = 0; i < N; ++i) {
        float frac = static_cast<float>(i) / static_cast<float>(N);
        gap[i] = 0.85f;
        cav[i] = 0.80f;
        blk[i] = (frac > 0.4f && frac < 0.7f) ? (frac - 0.4f) / 0.3f * 0.90f : 0.0f; // Cav3 blocker
        dbs[i] = (frac >= 0.7f) ? 1.0f : 0.0f; // VIM DBS
    }

    // Warmup
    compute_essential_tremor_avx2(
        gap.data(), cav.data(), blk.data(), dbs.data(),
        freq.data(), coh.data(), pow.data(), amp.data(), N
    );

    const int WARMUP_RUNS = 20;
    const int BENCH_RUNS = 100;
    for (int r = 0; r < WARMUP_RUNS; ++r) {
        compute_essential_tremor_avx2(
            gap.data(), cav.data(), blk.data(), dbs.data(),
            freq.data(), coh.data(), pow.data(), amp.data(), N
        );
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < BENCH_RUNS; ++r) {
        compute_essential_tremor_avx2(
            gap.data(), cav.data(), blk.data(), dbs.data(),
            freq.data(), coh.data(), pow.data(), amp.data(), N
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / BENCH_RUNS;
    double throughput_melem = (static_cast<double>(N) / (avg_ms / 1000.0)) / 1e6;

    std::cout << "=================================================================\n";
    std::cout << "WEFT AVX2/FMA Benchmark: Target #402 Essential Tremor & DRT\n";
    std::cout << "=================================================================\n";
    std::cout << "Vector Size:           " << N << " elements\n";
    std::cout << "Average Latency:       " << avg_ms << " ms per million state evaluations\n";
    std::cout << "Throughput:            " << throughput_melem << " Melem/s\n";
    std::cout << "Sample Trace:\n";
    std::cout << "  [Severe ET]         Freq: " << freq[1000] << " Hz | Coherence: " << coh[1000]*100.0f << "% | Tremor: " << amp[1000] << " cm\n";
    std::cout << "  [Cav3 Blocked]      Freq: " << freq[650000] << " Hz | Coherence: " << coh[650000]*100.0f << "% | Tremor: " << amp[650000] << " cm\n";
    std::cout << "  [VIM DBS Desync]    Freq: " << freq[850000] << " Hz | Coherence: " << coh[850000]*100.0f << "% | Tremor: " << amp[850000] << " cm\n";
    std::cout << "=================================================================\n";

    return 0;
}
