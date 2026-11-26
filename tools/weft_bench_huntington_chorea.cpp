#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

/**
 * WEFT AVX2 / FMA Microbenchmark: Milestone Target #400
 * Huntington's Disease: Striatal D2-SPN Degeneration & Hyperkinetic Chorea
 * Collaborative Research Target with Kevin W. McCairn, Ph.D.
 * 
 * Computes:
 * - GPe prototypic disinhibition under D2-SPN cell death
 * - STN motor brake suppression
 * - Thalamocortical motor disinhibition & UHDRS chorea score
 * - Pharmacological GABAergic rescue
 */

void compute_huntington_chorea_avx2(
    const float* __restrict__ d2_spn_survival,
    const float* __restrict__ gaba_rescue,
    float* __restrict__ gpe_rate,
    float* __restrict__ stn_rate,
    float* __restrict__ thalamus_drive,
    float* __restrict__ chorea_score,
    size_t n
) {
    const __m256 v_ones = _mm256_set1_ps(1.0f);
    const __m256 v_zeros = _mm256_setzero_ps();
    const __m256 v_gpe_base = _mm256_set1_ps(52.0f);
    const __m256 v_gpe_scale = _mm256_set1_ps(48.0f);
    const __m256 v_rescue_scale = _mm256_set1_ps(0.75f);
    const __m256 v_gpe_thresh = _mm256_set1_ps(50.0f);
    const __m256 v_stn_base = _mm256_set1_ps(24.0f);
    const __m256 v_stn_scale = _mm256_set1_ps(0.40f);
    const __m256 v_stn_min = _mm256_set1_ps(2.0f);
    const __m256 v_stn_ref = _mm256_set1_ps(25.0f);
    const __m256 v_thal_base = _mm256_set1_ps(20.0f);
    const __m256 v_thal_scale = _mm256_set1_ps(3.8f);
    const __m256 v_thal_max = _mm256_set1_ps(100.0f);
    const __m256 v_thal_thresh = _mm256_set1_ps(40.0f);
    const __m256 v_chorea_scale = _mm256_set1_ps(1.0f / 5.5f);
    const __m256 v_chorea_max = _mm256_set1_ps(10.0f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_surv = _mm256_loadu_ps(&d2_spn_survival[i]);
        __m256 v_resc = _mm256_loadu_ps(&gaba_rescue[i]);

        v_surv = _mm256_max_ps(v_zeros, _mm256_min_ps(v_ones, v_surv));
        v_resc = _mm256_max_ps(v_zeros, _mm256_min_ps(v_ones, v_resc));

        __m256 d2_loss = _mm256_sub_ps(v_ones, v_surv);
        __m256 resc_factor = _mm256_sub_ps(v_ones, _mm256_mul_ps(v_resc, v_rescue_scale));

        // GPe = 52.0 + 48.0 * d2_loss * resc_factor
        __m256 gpe = _mm256_fmadd_ps(v_gpe_scale, _mm256_mul_ps(d2_loss, resc_factor), v_gpe_base);
        _mm256_storeu_ps(&gpe_rate[i], gpe);

        // STN = max(2.0, 24.0 - 0.40 * max(0.0, gpe - 50.0))
        __m256 gpe_excess = _mm256_max_ps(v_zeros, _mm256_sub_ps(gpe, v_gpe_thresh));
        __m256 stn = _mm256_max_ps(v_stn_min, _mm256_fnmadd_ps(v_stn_scale, gpe_excess, v_stn_base));
        _mm256_storeu_ps(&stn_rate[i], stn);

        // Thalamus = min(100.0, max(10.0, 20.0 + 3.8 * max(0.0, 25.0 - stn)))
        __m256 stn_deficit = _mm256_max_ps(v_zeros, _mm256_sub_ps(v_stn_ref, stn));
        __m256 thal = _mm256_min_ps(v_thal_max, _mm256_fmadd_ps(v_thal_scale, stn_deficit, v_thal_base));
        _mm256_storeu_ps(&thalamus_drive[i], thal);

        // Chorea = min(10.0, max(0.0, (thal - 40.0) / 5.5))
        __m256 thal_excess = _mm256_max_ps(v_zeros, _mm256_sub_ps(thal, v_thal_thresh));
        __m256 chorea = _mm256_min_ps(v_chorea_max, _mm256_mul_ps(thal_excess, v_chorea_scale));
        _mm256_storeu_ps(&chorea_score[i], chorea);
    }
}

int main() {
    const size_t N = 1000000; // 1M state samples
    std::vector<float> d2_survival(N), gaba_rescue(N);
    std::vector<float> gpe_rate(N), stn_rate(N), thalamus_drive(N), chorea_score(N);

    // Populate disease trajectory
    for (size_t i = 0; i < N; ++i) {
        float frac = static_cast<float>(i) / static_cast<float>(N);
        d2_survival[i] = (frac < 0.6f) ? (1.0f - frac / 0.6f * 0.85f) : 0.15f; // Degenerates to 15%
        gaba_rescue[i] = (frac > 0.8f) ? (frac - 0.8f) / 0.2f * 0.90f : 0.0f; // Late rescue infusion
    }

    // Warmup
    compute_huntington_chorea_avx2(
        d2_survival.data(), gaba_rescue.data(), gpe_rate.data(),
        stn_rate.data(), thalamus_drive.data(), chorea_score.data(), N
    );

    const int WARMUP_RUNS = 20;
    const int BENCH_RUNS = 100;
    for (int r = 0; r < WARMUP_RUNS; ++r) {
        compute_huntington_chorea_avx2(
            d2_survival.data(), gaba_rescue.data(), gpe_rate.data(),
            stn_rate.data(), thalamus_drive.data(), chorea_score.data(), N
        );
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < BENCH_RUNS; ++r) {
        compute_huntington_chorea_avx2(
            d2_survival.data(), gaba_rescue.data(), gpe_rate.data(),
            stn_rate.data(), thalamus_drive.data(), chorea_score.data(), N
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / BENCH_RUNS;
    double throughput_melem = (static_cast<double>(N) / (avg_ms / 1000.0)) / 1e6;

    std::cout << "=================================================================\n";
    std::cout << "WEFT AVX2/FMA Benchmark: Milestone Target #400 Huntington Chorea\n";
    std::cout << "=================================================================\n";
    std::cout << "Vector Size:           " << N << " elements\n";
    std::cout << "Average Latency:       " << avg_ms << " ms per million state evaluations\n";
    std::cout << "Throughput:            " << throughput_melem << " Melem/s\n";
    std::cout << "Sample Trace:\n";
    std::cout << "  [Healthy Baseline]  D2-SPN: " << d2_survival[0]*100.0f << "% | GPe: " << gpe_rate[0] << " Hz | STN Brake: " << stn_rate[0] << " Hz | Chorea: " << chorea_score[0] << "/10\n";
    std::cout << "  [Severe Chorea]     D2-SPN: " << d2_survival[700000]*100.0f << "% | GPe: " << gpe_rate[700000] << " Hz | STN Brake: " << stn_rate[700000] << " Hz | Chorea: " << chorea_score[700000] << "/10\n";
    std::cout << "  [GABA Rescued]      Rescue: " << gaba_rescue[950000]*100.0f << "% | GPe: " << gpe_rate[950000] << " Hz | STN Brake: " << stn_rate[950000] << " Hz | Chorea: " << chorea_score[950000] << "/10\n";
    std::cout << "=================================================================\n";

    return 0;
}
