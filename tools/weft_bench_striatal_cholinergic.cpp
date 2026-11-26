#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

/**
 * WEFT AVX2 / FMA Microbenchmark: Target #398
 * Striatal Cholinergic Interneuron (ChI) Hyper-Rebound & Dystonic Co-Contraction
 * Collaborative Research Target with Kevin W. McCairn, Ph.D.
 * 
 * Computes:
 * - ChI TAN firing rates under Dopamine D2 tone attenuation
 * - Striatal Acetylcholine [ACh] overflow
 * - Agonist/Antagonist motor drive and Co-Contraction Index (CCI)
 */

void compute_cholinergic_dystonia_avx2(
    const float* __restrict__ d2_tone,
    const float* __restrict__ m4_rescue,
    float* __restrict__ chi_rate,
    float* __restrict__ ach_conc,
    float* __restrict__ agonist_drive,
    float* __restrict__ antagonist_leak,
    float* __restrict__ cci,
    size_t n
) {
    const __m256 v_ones = _mm256_set1_ps(1.0f);
    const __m256 v_zeros = _mm256_setzero_ps();
    const __m256 v_chi_baseline = _mm256_set1_ps(4.5f);
    const __m256 v_rebound_scale = _mm256_set1_ps(25.0f);
    const __m256 v_m4_suppr_scale = _mm256_set1_ps(0.70f);
    const __m256 v_ach_scale = _mm256_set1_ps(4.2f);
    const __m256 v_ag_base = _mm256_set1_ps(30.0f);
    const __m256 v_ag_scale = _mm256_set1_ps(0.25f);
    const __m256 v_ant_base = _mm256_set1_ps(8.0f);
    const __m256 v_ant_scale = _mm256_set1_ps(0.20f);
    const __m256 v_ant_m4_scale = _mm256_set1_ps(0.85f);
    const __m256 v_two = _mm256_set1_ps(2.0f);
    const __m256 v_d2_thresh = _mm256_set1_ps(0.40f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_d2 = _mm256_loadu_ps(&d2_tone[i]);
        __m256 v_m4 = _mm256_loadu_ps(&m4_rescue[i]);

        // Bound inputs [0.0, 1.0]
        v_d2 = _mm256_max_ps(v_zeros, _mm256_min_ps(v_ones, v_d2));
        v_m4 = _mm256_max_ps(v_zeros, _mm256_min_ps(v_ones, v_m4));

        __m256 d2_loss = _mm256_sub_ps(v_ones, v_d2);
        __m256 m4_suppr = _mm256_sub_ps(v_ones, _mm256_mul_ps(v_m4, v_m4_suppr_scale));

        // chi_rate = baseline + 25.0 * d2_loss * m4_suppr
        __m256 rebound = _mm256_mul_ps(v_rebound_scale, _mm256_mul_ps(d2_loss, m4_suppr));
        __m256 rate = _mm256_add_ps(v_chi_baseline, rebound);
        _mm256_storeu_ps(&chi_rate[i], rate);

        // [ACh] = rate * 4.2
        __m256 ach = _mm256_mul_ps(rate, v_ach_scale);
        _mm256_storeu_ps(&ach_conc[i], ach);

        // Agonist drive = 30.0 + 0.25 * rate
        __m256 ag = _mm256_fmadd_ps(rate, v_ag_scale, v_ag_base);
        _mm256_storeu_ps(&agonist_drive[i], ag);

        // Antagonist leakage = 8.0 + (if d2 < 0.4 then rate * 0.20 * (1 - 0.85 * m4) else 0)
        __m256 d2_mask = _mm256_cmp_ps(v_d2, v_d2_thresh, _CMP_LT_OQ);
        __m256 ant_m4_factor = _mm256_sub_ps(v_ones, _mm256_mul_ps(v_m4, v_ant_m4_scale));
        __m256 ant_extra = _mm256_mul_ps(rate, _mm256_mul_ps(v_ant_scale, ant_m4_factor));
        ant_extra = _mm256_and_ps(d2_mask, ant_extra);
        __m256 ant = _mm256_add_ps(v_ant_base, ant_extra);
        _mm256_storeu_ps(&antagonist_leak[i], ant);

        // Co-contraction index = 2.0 * min(ag, ant) / (ag + ant)
        __m256 min_act = _mm256_min_ps(ag, ant);
        __m256 sum_act = _mm256_add_ps(ag, ant);
        __m256 cci_val = _mm256_div_ps(_mm256_mul_ps(v_two, min_act), sum_act);
        _mm256_storeu_ps(&cci[i], cci_val);
    }
}

int main() {
    const size_t N = 1000000; // 1M state samples
    std::vector<float> d2_tone(N), m4_rescue(N);
    std::vector<float> chi_rate(N), ach_conc(N), agonist_drive(N), antagonist_leak(N), cci(N);

    // Populate synthetic trace
    for (size_t i = 0; i < N; ++i) {
        float frac = static_cast<float>(i) / static_cast<float>(N);
        d2_tone[i] = (frac < 0.5f) ? (1.0f - 2.0f * frac) : 0.05f; // Progressive dopamine depletion
        m4_rescue[i] = (frac > 0.75f) ? (frac - 0.75f) * 4.0f : 0.0f; // Late rescue agent infusion
    }

    // Warmup
    compute_cholinergic_dystonia_avx2(
        d2_tone.data(), m4_rescue.data(), chi_rate.data(), ach_conc.data(),
        agonist_drive.data(), antagonist_leak.data(), cci.data(), N
    );

    // Benchmarking loop
    const int WARMUP_RUNS = 20;
    const int BENCH_RUNS = 100;
    for (int r = 0; r < WARMUP_RUNS; ++r) {
        compute_cholinergic_dystonia_avx2(
            d2_tone.data(), m4_rescue.data(), chi_rate.data(), ach_conc.data(),
            agonist_drive.data(), antagonist_leak.data(), cci.data(), N
        );
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < BENCH_RUNS; ++r) {
        compute_cholinergic_dystonia_avx2(
            d2_tone.data(), m4_rescue.data(), chi_rate.data(), ach_conc.data(),
            agonist_drive.data(), antagonist_leak.data(), cci.data(), N
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / BENCH_RUNS;
    double throughput_melem = (static_cast<double>(N) / (avg_ms / 1000.0)) / 1e6;

    std::cout << "=================================================================\n";
    std::cout << "WEFT AVX2/FMA Benchmark: Target #398 Striatal Cholinergic Dystonia\n";
    std::cout << "=================================================================\n";
    std::cout << "Vector Size:           " << N << " elements\n";
    std::cout << "Average Latency:       " << avg_ms << " ms per million state evaluations\n";
    std::cout << "Throughput:            " << throughput_melem << " Melem/s\n";
    std::cout << "Sample Output (idx 0 - Healthy Baseline):\n";
    std::cout << "  D2 Tone:             " << d2_tone[0] << "\n";
    std::cout << "  ChI Rate:            " << chi_rate[0] << " Hz\n";
    std::cout << "  [ACh]:               " << ach_conc[0] << " nM\n";
    std::cout << "  Co-Contraction (CCI):" << cci[0] * 100.0f << " %\n";
    std::cout << "Sample Output (idx 600000 - Depleted / Dystonic):\n";
    std::cout << "  D2 Tone:             " << d2_tone[600000] << "\n";
    std::cout << "  ChI Rate:            " << chi_rate[600000] << " Hz\n";
    std::cout << "  [ACh]:               " << ach_conc[600000] << " nM\n";
    std::cout << "  Co-Contraction (CCI):" << cci[600000] * 100.0f << " %\n";
    std::cout << "Sample Output (idx 950000 - M4 PAM Rescued):\n";
    std::cout << "  M4 Rescue:           " << m4_rescue[950000] << "\n";
    std::cout << "  ChI Rate:            " << chi_rate[950000] << " Hz\n";
    std::cout << "  Co-Contraction (CCI):" << cci[950000] * 100.0f << " %\n";
    std::cout << "=================================================================\n";

    return 0;
}
