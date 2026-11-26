// Vectorized microbenchmark for Thalamic Reticular Nucleus (TRN) Attentional Gating
// Evaluates AVX2 + FMA SIMD throughput of cross-modal lateral inhibition (Visual LGN vs Auditory MGN) and burst-tonic mode gating.
// Part of the WEFT Monorepo benchmark suite (Target #392).

#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

struct alignas(32) TRNBatch {
    static constexpr size_t N = 1000000;
    std::vector<float> visual_input;
    std::vector<float> auditory_input;
    std::vector<float> pfc_visual_bias;
    std::vector<float> membrane_potential;
    std::vector<float> cholinergic_arousal;
    std::vector<float> output_vis_relay;
    std::vector<float> output_aud_relay;
    std::vector<float> output_trn_vis_gaba;
    std::vector<float> output_trn_aud_gaba;

    TRNBatch()
        : visual_input(N), auditory_input(N), pfc_visual_bias(N), membrane_potential(N), cholinergic_arousal(N),
          output_vis_relay(N), output_aud_relay(N), output_trn_vis_gaba(N), output_trn_aud_gaba(N) {
        for (size_t i = 0; i < N; ++i) {
            visual_input[i] = 2.0f + (static_cast<float>(i % 40) / 10.0f);
            auditory_input[i] = 2.0f + (static_cast<float>((i + 5) % 40) / 10.0f);
            pfc_visual_bias[i] = -0.8f + (static_cast<float>(i % 33) / 20.0f); // -0.8 to +0.8
            membrane_potential[i] = -70.0f + (static_cast<float>(i % 25)); // -70 to -45 mV
            cholinergic_arousal[i] = 1.0f + (static_cast<float>(i % 20) / 5.0f);
        }
    }
};

void benchmark_trn_filtering_avx2(TRNBatch& batch) {
    const size_t n = TRNBatch::N;
    const float* __restrict vis_in = batch.visual_input.data();
    const float* __restrict aud_in = batch.auditory_input.data();
    const float* __restrict pfc = batch.pfc_visual_bias.data();
    const float* __restrict vm = batch.membrane_potential.data();
    const float* __restrict ach = batch.cholinergic_arousal.data();
    float* __restrict out_vis = batch.output_vis_relay.data();
    float* __restrict out_aud = batch.output_aud_relay.data();
    float* __restrict out_vgaba = batch.output_trn_vis_gaba.data();
    float* __restrict out_agaba = batch.output_trn_aud_gaba.data();

    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_base_trn = _mm256_set1_ps(2.0f);
    const __m256 v_pfc_gain = _mm256_set1_ps(3.5f);
    const __m256 v_scale_arousal = _mm256_set1_ps(1.5f);
    const __m256 v_burst_thresh = _mm256_set1_ps(-65.0f);
    const __m256 v_burst_val = _mm256_set1_ps(8.0f);
    const __m256 v_tonic_gain = _mm256_set1_ps(1.5f);
    const __m256 v_gaba_inh = _mm256_set1_ps(1.2f);
    const __m256 v_half = _mm256_set1_ps(0.5f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_vin = _mm256_loadu_ps(vis_in + i);
        __m256 v_ain = _mm256_loadu_ps(aud_in + i);
        __m256 v_pfc_val = _mm256_loadu_ps(pfc + i);
        __m256 v_vm_val = _mm256_loadu_ps(vm + i);
        __m256 v_ach_val = _mm256_loadu_ps(ach + i);

        // visGABA = max(0, baseTRN - pfc * pfcGain)
        __m256 v_pfc_bias = _mm256_mul_ps(v_pfc_val, v_pfc_gain);
        __m256 v_vgaba = _mm256_max_ps(v_zero, _mm256_sub_ps(v_base_trn, v_pfc_bias));

        // audGABA = max(0, baseTRN + pfc * pfcGain)
        __m256 v_agaba = _mm256_max_ps(v_zero, _mm256_add_ps(v_base_trn, v_pfc_bias));

        // effectiveVm = Vm + 1.5 * ACh
        __m256 v_eff_vm = _mm256_fmadd_ps(v_scale_arousal, v_ach_val, v_vm_val);

        // mask_burst = effectiveVm < -65.0 mV
        __m256 v_mask_burst = _mm256_cmp_ps(v_eff_vm, v_burst_thresh, _CMP_LT_OQ);

        // Tonic output:
        // visOut_tonic = max(0, vin * 1.5 - vgaba * 1.2)
        __m256 v_vout_tonic = _mm256_max_ps(v_zero, _mm256_fmsub_ps(v_vin, v_tonic_gain, _mm256_mul_ps(v_vgaba, v_gaba_inh)));
        __m256 v_aout_tonic = _mm256_max_ps(v_zero, _mm256_fmsub_ps(v_ain, v_tonic_gain, _mm256_mul_ps(v_agaba, v_gaba_inh)));

        // Burst output:
        // visOut_burst = max(0, 8.0 - vgaba * 0.5)
        __m256 v_vout_burst = _mm256_max_ps(v_zero, _mm256_fmsub_ps(v_vgaba, v_half, v_burst_val));
        __m256 v_aout_burst = _mm256_max_ps(v_zero, _mm256_fmsub_ps(v_agaba, v_half, v_burst_val));

        // Blend based on burst mask
        __m256 v_vout = _mm256_blendv_ps(v_vout_tonic, v_vout_burst, v_mask_burst);
        __m256 v_aout = _mm256_blendv_ps(v_aout_tonic, v_aout_burst, v_mask_burst);

        _mm256_storeu_ps(out_vis + i, v_vout);
        _mm256_storeu_ps(out_aud + i, v_aout);
        _mm256_storeu_ps(out_vgaba + i, v_vgaba);
        _mm256_storeu_ps(out_agaba + i, v_agaba);
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "WEFT AVX2+FMA Thalamic Reticular Gating Benchmark (T392)\n";
    std::cout << "========================================================\n";

    TRNBatch batch;

    // Warmup
    for (int w = 0; w < 5; ++w) {
        benchmark_trn_filtering_avx2(batch);
    }

    constexpr int ITERS = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        benchmark_trn_filtering_avx2(batch);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / ITERS;
    double throughput = (static_cast<double>(TRNBatch::N) / (avg_ms / 1000.0)) / 1e6;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Batch Size:           " << TRNBatch::N << " elements\n";
    std::cout << "Total Runtime (" << ITERS << "x):   " << total_ms << " ms\n";
    std::cout << "Latency per batch:    " << avg_ms << " ms\n";
    std::cout << "Throughput:           " << throughput << " Melem/s\n";

    // Verification sample
    std::cout << "\nSample Verification at index 0 (VisIn=" << batch.visual_input[0] 
              << ", AudIn=" << batch.auditory_input[0] << ", PFC=" << batch.pfc_visual_bias[0] << "):\n";
    std::cout << "  Visual TRN GABA:   " << batch.output_trn_vis_gaba[0] << "\n";
    std::cout << "  Auditory TRN GABA: " << batch.output_trn_aud_gaba[0] << "\n";
    std::cout << "  Visual Relay Out:  " << batch.output_vis_relay[0] << "\n";
    std::cout << "  Auditory Relay Out:" << batch.output_aud_relay[0] << "\n";
    std::cout << "========================================================\n";

    return 0;
}
