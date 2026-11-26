/**
 * tools/weft_bench_lh_orexin.cpp
 * AVX2/FMA Native Vectorized SIMD Microbenchmark
 * Lateral Hypothalamus (LH) Orexin Arousal & Muscle Tone Gate
 * Lean 4 Formal Target #396 | Paper #205 | Cycle v340.74.0
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>
#include <iomanip>

void run_lh_orexin_avx2_fma(
    const float* __restrict__ valence,
    const float* __restrict__ integrity,
    const float* __restrict__ circadian,
    float* __restrict__ out_orexin_rate,
    float* __restrict__ out_mono_tone,
    float* __restrict__ out_sld_atonia,
    float* __restrict__ out_muscle_tone,
    float* __restrict__ out_cataplexy_prob,
    size_t count
) {
    const __m256 c_zero        = _mm256_setzero_ps();
    const __m256 c_one         = _mm256_set1_ps(1.0f);
    const __m256 c_ox_base     = _mm256_set1_ps(3.0f);
    const __m256 c_ox_val      = _mm256_set1_ps(25.0f);
    const __m256 c_ox_circ     = _mm256_set1_ps(12.0f);
    const __m256 c_ox_max      = _mm256_set1_ps(45.0f);

    const __m256 c_mono_base   = _mm256_set1_ps(0.15f);
    const __m256 c_mono_scale  = _mm256_set1_ps(0.02f);

    const __m256 c_sld_val     = _mm256_set1_ps(0.90f);
    const __m256 c_sld_mono_inh= _mm256_set1_ps(0.75f);

    const __m256 c_tone_mono   = _mm256_set1_ps(0.90f);
    const __m256 c_tone_atonia = _mm256_set1_ps(1.10f);

    const __m256 c_cata_th_atonia = _mm256_set1_ps(0.50f);
    const __m256 c_cata_th_tone   = _mm256_set1_ps(0.20f);
    const __m256 c_cata_slope     = _mm256_set1_ps(2.5f);

    for (size_t i = 0; i < count; i += 8) {
        __m256 v_val = _mm256_loadu_ps(valence + i);
        __m256 v_int = _mm256_loadu_ps(integrity + i);
        __m256 v_circ = _mm256_loadu_ps(circadian + i);

        // 1. LH Orexin Firing Rate: clamp((3.0 + 25.0*val + 12.0*circ)*int, 0.0, 45.0)
        __m256 v_raw_ox = _mm256_fmadd_ps(c_ox_val, v_val, c_ox_base);
        v_raw_ox = _mm256_fmadd_ps(c_ox_circ, v_circ, v_raw_ox);
        __m256 v_ox = _mm256_mul_ps(v_raw_ox, v_int);
        v_ox = _mm256_max_ps(c_zero, _mm256_min_ps(c_ox_max, v_ox));

        // 2. Monoaminergic Tone: clamp(0.15 + 0.02 * ox, 0.0, 1.0)
        __m256 v_mono = _mm256_fmadd_ps(c_mono_scale, v_ox, c_mono_base);
        v_mono = _mm256_max_ps(c_zero, _mm256_min_ps(c_one, v_mono));

        // 3. SLD Atonia Drive: clamp(0.90 * val - 0.75 * mono, 0.0, 1.0)
        __m256 v_sld = _mm256_mul_ps(c_sld_val, v_val);
        v_sld = _mm256_fnmadd_ps(c_sld_mono_inh, v_mono, v_sld);
        v_sld = _mm256_max_ps(c_zero, _mm256_min_ps(c_one, v_sld));

        // 4. Somatic Muscle Tone: clamp(0.90 * mono - 1.10 * sld, 0.0, 1.0)
        __m256 v_tone = _mm256_mul_ps(c_tone_mono, v_mono);
        v_tone = _mm256_fnmadd_ps(c_tone_atonia, v_sld, v_tone);
        v_tone = _mm256_max_ps(c_zero, _mm256_min_ps(c_one, v_tone));

        // 5. Cataplexy Probability: clamp(cata_slope * (sld - 0.50) * (0.20 - tone), 0.0, 1.0)
        __m256 v_diff_atonia = _mm256_max_ps(c_zero, _mm256_sub_ps(v_sld, c_cata_th_atonia));
        __m256 v_diff_tone = _mm256_max_ps(c_zero, _mm256_sub_ps(c_cata_th_tone, v_tone));
        __m256 v_cata = _mm256_mul_ps(_mm256_mul_ps(v_diff_atonia, v_diff_tone), _mm256_set1_ps(25.0f));
        v_cata = _mm256_max_ps(c_zero, _mm256_min_ps(c_one, v_cata));

        _mm256_storeu_ps(out_orexin_rate + i, v_ox);
        _mm256_storeu_ps(out_mono_tone + i, v_mono);
        _mm256_storeu_ps(out_sld_atonia + i, v_sld);
        _mm256_storeu_ps(out_muscle_tone + i, v_tone);
        _mm256_storeu_ps(out_cataplexy_prob + i, v_cata);
    }
}

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "WEFT AVX2/FMA SIMD Microbenchmark: LH Orexin & Muscle Tone Gate" << std::endl;
    std::cout << "Lean 4 Target #396 | Paper #205 | Cycle v340.74.0" << std::endl;
    std::cout << "================================================================================" << std::endl;

    const size_t N = 1000000;
    std::vector<float> val(N, 0.85f);
    std::vector<float> integrity(N, 1.00f);
    std::vector<float> circ(N, 0.75f);

    std::vector<float> out_ox(N);
    std::vector<float> out_mono(N);
    std::vector<float> out_sld(N);
    std::vector<float> out_tone(N);
    std::vector<float> out_cata(N);

    // Warmup
    for (int w = 0; w < 5; ++w) {
        run_lh_orexin_avx2_fma(val.data(), integrity.data(), circ.data(),
                               out_ox.data(), out_mono.data(), out_sld.data(),
                               out_tone.data(), out_cata.data(), N);
    }

    // Benchmark
    const int RUNS = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        run_lh_orexin_avx2_fma(val.data(), integrity.data(), circ.data(),
                               out_ox.data(), out_mono.data(), out_sld.data(),
                               out_tone.data(), out_cata.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / RUNS;
    double throughput = (double)N / (avg_ms * 1e-3) / 1e6;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Processed Elements:  " << N << " (8-lane AVX2/FMA SIMD)" << std::endl;
    std::cout << "Average Latency:     " << avg_ms << " ms" << std::endl;
    std::cout << "Peak Throughput:     " << throughput << " Melem/s (" << (throughput / 1e3) << " Gelem/s)" << std::endl;
    std::cout << "\nSample Verification at index 0 (Healthy Intact State):" << std::endl;
    std::cout << "  Emotional Valence:   " << val[0] * 100.0f << "%" << std::endl;
    std::cout << "  Orexin Integrity:    " << integrity[0] * 100.0f << "%" << std::endl;
    std::cout << "  Circadian Drive:     " << circ[0] * 100.0f << "%" << std::endl;
    std::cout << "  Orexin Firing Rate:  " << out_ox[0] << " Hz" << std::endl;
    std::cout << "  Monoaminergic Tone:  " << out_mono[0] * 100.0f << "%" << std::endl;
    std::cout << "  SLD Atonia Drive:    " << out_sld[0] * 100.0f << "%" << std::endl;
    std::cout << "  Somatic Muscle Tone: " << out_tone[0] * 100.0f << "%" << std::endl;
    std::cout << "  Cataplexy Collapse:  " << out_cata[0] * 100.0f << "%" << std::endl;

    // Run narcoleptic trial
    std::vector<float> narc_int(8, 0.0f);
    std::vector<float> narc_val(8, 0.85f);
    std::vector<float> narc_circ(8, 0.75f);
    std::vector<float> narc_ox(8), narc_mono(8), narc_sld(8), narc_tone(8), narc_cata(8);
    run_lh_orexin_avx2_fma(narc_val.data(), narc_int.data(), narc_circ.data(),
                           narc_ox.data(), narc_mono.data(), narc_sld.data(),
                           narc_tone.data(), narc_cata.data(), 8);

    std::cout << "\nSample Verification at index 0 (Narcoleptic Degenerated State, Integrity = 0%):" << std::endl;
    std::cout << "  Orexin Firing Rate:  " << narc_ox[0] << " Hz (Abolished)" << std::endl;
    std::cout << "  Monoaminergic Tone:  " << narc_mono[0] * 100.0f << "%" << std::endl;
    std::cout << "  SLD Atonia Drive:    " << narc_sld[0] * 100.0f << "% (Disinhibited)" << std::endl;
    std::cout << "  Somatic Muscle Tone: " << narc_tone[0] * 100.0f << "% (Atonia collapse)" << std::endl;
    std::cout << "  Cataplexy Collapse:  " << narc_cata[0] * 100.0f << "% (Cataplexy triggered)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    return 0;
}
