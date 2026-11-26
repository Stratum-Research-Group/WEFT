#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

/**
 * WEFT AVX2 / FMA Microbenchmark: Target #401
 * ALS: Corticomotoneuronal Excitotoxicity & Renshaw Recurrent Disinhibition
 * Collaborative Research Target with Kevin W. McCairn, Ph.D.
 * 
 * Computes:
 * - Astrocytic EAAT2 loss & synaptic glutamate accumulation
 * - Renshaw recurrent glycinergic inhibition gain
 * - Betz / alpha motoneuron excitotoxic firing rate
 * - Spontaneous fasciculation potential frequency & spasticity index
 */

void compute_als_excitotoxicity_avx2(
    const float* __restrict__ eaat2_level,
    const float* __restrict__ renshaw_integ,
    const float* __restrict__ riluzole_block,
    const float* __restrict__ glycine_boost,
    float* __restrict__ synaptic_glu,
    float* __restrict__ motoneuron_rate,
    float* __restrict__ renshaw_gain,
    float* __restrict__ fasciculation_freq,
    size_t n
) {
    const __m256 v_ones = _mm256_set1_ps(1.0f);
    const __m256 v_zeros = _mm256_setzero_ps();
    const __m256 v_glu_base = _mm256_set1_ps(1.2f);
    const __m256 v_glu_scale = _mm256_set1_ps(18.0f);
    const __m256 v_ril_scale = _mm256_set1_ps(0.70f);
    const __m256 v_ren_r_scale = _mm256_set1_ps(0.80f);
    const __m256 v_ren_g_scale = _mm256_set1_ps(0.50f);
    const __m256 v_moto_base = _mm256_set1_ps(12.0f);
    const __m256 v_moto_glu_scale = _mm256_set1_ps(3.5f);
    const __m256 v_moto_ren_scale = _mm256_set1_ps(24.0f);
    const __m256 v_moto_min = _mm256_set1_ps(8.0f);
    const __m256 v_glu_thresh = _mm256_set1_ps(4.0f);
    const __m256 v_ren_thresh = _mm256_set1_ps(0.50f);
    const __m256 v_fasc_scale = _mm256_set1_ps(0.45f);
    const __m256 v_fasc_base = _mm256_set1_ps(0.1f);
    const __m256 v_fasc_max = _mm256_set1_ps(8.0f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_eaat2 = _mm256_max_ps(v_zeros, _mm256_min_ps(v_ones, _mm256_loadu_ps(&eaat2_level[i])));
        __m256 v_ren = _mm256_max_ps(v_zeros, _mm256_min_ps(v_ones, _mm256_loadu_ps(&renshaw_integ[i])));
        __m256 v_ril = _mm256_max_ps(v_zeros, _mm256_min_ps(v_ones, _mm256_loadu_ps(&riluzole_block[i])));
        __m256 v_gly = _mm256_max_ps(v_zeros, _mm256_min_ps(v_ones, _mm256_loadu_ps(&glycine_boost[i])));

        // Synaptic glutamate = 1.2 + 18.0 * (1 - eaat2) * (1 - 0.70 * riluzole)
        __m256 eaat2_loss = _mm256_sub_ps(v_ones, v_eaat2);
        __m256 ril_factor = _mm256_sub_ps(v_ones, _mm256_mul_ps(v_ril, v_ril_scale));
        __m256 glu = _mm256_fmadd_ps(v_glu_scale, _mm256_mul_ps(eaat2_loss, ril_factor), v_glu_base);
        _mm256_storeu_ps(&synaptic_glu[i], glu);

        // Renshaw gain = min(1.0, 0.80 * ren + 0.50 * gly)
        __m256 ren_gain = _mm256_min_ps(v_ones, _mm256_fmadd_ps(v_ren, v_ren_r_scale, _mm256_mul_ps(v_gly, v_ren_g_scale)));
        _mm256_storeu_ps(&renshaw_gain[i], ren_gain);

        // Motoneuron rate = max(8.0, 12.0 + 3.5 * glu - 24.0 * ren_gain)
        __m256 raw_rate = _mm256_fmadd_ps(glu, v_moto_glu_scale, v_moto_base);
        __m256 rate = _mm256_max_ps(v_moto_min, _mm256_fnmadd_ps(ren_gain, v_moto_ren_scale, raw_rate));
        _mm256_storeu_ps(&motoneuron_rate[i], rate);

        // Fasciculation freq: if glu > 4.0 and ren_gain < 0.5
        __m256 glu_cond = _mm256_cmp_ps(glu, v_glu_thresh, _CMP_GT_OQ);
        __m256 ren_cond = _mm256_cmp_ps(ren_gain, v_ren_thresh, _CMP_LT_OQ);
        __m256 fasc_mask = _mm256_and_ps(glu_cond, ren_cond);

        __m256 glu_excess = _mm256_sub_ps(glu, v_glu_thresh);
        __m256 ren_deficit = _mm256_sub_ps(v_ones, ren_gain);
        __m256 raw_fasc = _mm256_min_ps(v_fasc_max, _mm256_mul_ps(v_fasc_scale, _mm256_mul_ps(glu_excess, ren_deficit)));
        __m256 fasc = _mm256_blendv_ps(v_fasc_base, raw_fasc, fasc_mask);
        _mm256_storeu_ps(&fasciculation_freq[i], fasc);
    }
}

int main() {
    const size_t N = 1000000;
    std::vector<float> eaat2(N), renshaw(N), riluzole(N), glycine(N);
    std::vector<float> glu(N), moto_rate(N), ren_gain(N), fasc(N);

    // Populate disease trajectory
    for (size_t i = 0; i < N; ++i) {
        float frac = static_cast<float>(i) / static_cast<float>(N);
        eaat2[i] = (frac < 0.5f) ? (1.0f - frac / 0.5f * 0.85f) : 0.15f;
        renshaw[i] = (frac < 0.5f) ? (1.0f - frac / 0.5f * 0.80f) : 0.20f;
        riluzole[i] = (frac > 0.75f) ? (frac - 0.75f) / 0.25f * 0.85f : 0.0f;
        glycine[i] = (frac > 0.75f) ? (frac - 0.75f) / 0.25f * 0.80f : 0.0f;
    }

    // Warmup
    compute_als_excitotoxicity_avx2(
        eaat2.data(), renshaw.data(), riluzole.data(), glycine.data(),
        glu.data(), moto_rate.data(), ren_gain.data(), fasc.data(), N
    );

    const int WARMUP_RUNS = 20;
    const int BENCH_RUNS = 100;
    for (int r = 0; r < WARMUP_RUNS; ++r) {
        compute_als_excitotoxicity_avx2(
            eaat2.data(), renshaw.data(), riluzole.data(), glycine.data(),
            glu.data(), moto_rate.data(), ren_gain.data(), fasc.data(), N
        );
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < BENCH_RUNS; ++r) {
        compute_als_excitotoxicity_avx2(
            eaat2.data(), renshaw.data(), riluzole.data(), glycine.data(),
            glu.data(), moto_rate.data(), ren_gain.data(), fasc.data(), N
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / BENCH_RUNS;
    double throughput_melem = (static_cast<double>(N) / (avg_ms / 1000.0)) / 1e6;

    std::cout << "=================================================================\n";
    std::cout << "WEFT AVX2/FMA Benchmark: Target #401 ALS Excitotoxicity & Renshaw\n";
    std::cout << "=================================================================\n";
    std::cout << "Vector Size:           " << N << " elements\n";
    std::cout << "Average Latency:       " << avg_ms << " ms per million state evaluations\n";
    std::cout << "Throughput:            " << throughput_melem << " Melem/s\n";
    std::cout << "Sample Trace:\n";
    std::cout << "  [Healthy Baseline]  EAAT2: " << eaat2[0]*100.0f << "% | [Glu]: " << glu[0] << " uM | Motoneuron: " << moto_rate[0] << " Hz | Fasc: " << fasc[0] << " Hz\n";
    std::cout << "  [ALS Excitotoxic]   EAAT2: " << eaat2[600000]*100.0f << "% | [Glu]: " << glu[600000] << " uM | Motoneuron: " << moto_rate[600000] << " Hz | Fasc: " << fasc[600000] << " Hz\n";
    std::cout << "  [Rescued Homeost]   Riluz: " << riluzole[950000]*100.0f << "% | [Glu]: " << glu[950000] << " uM | Motoneuron: " << moto_rate[950000] << " Hz | Fasc: " << fasc[950000] << " Hz\n";
    std::cout << "=================================================================\n";

    return 0;
}
