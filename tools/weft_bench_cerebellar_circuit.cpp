// tools/weft_bench_cerebellar_circuit.cpp
// AVX2 SIMD Cerebellar Granule-Purkinje-DCN Circuit Microbenchmark
// Stratum Research Group - Nicholas Charles Mynhardt

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>
#include <cstdlib>

void run_cerebellar_avx2_bench(size_t num_synapses, size_t steps) {
    std::vector<float> pf_activity(num_synapses, 0.8f);
    std::vector<float> pf_weights(num_synapses, 0.75f);
    std::vector<float> pc_output(num_synapses / 8, 0.0f); // Purkinje cell sum

    float cf_error = 1.0f; // Inferior olive climbing fiber error burst
    float eta_ltd = 0.05f;
    float w_min = 0.05f;

    __m256 v_eta_ltd = _mm256_set1_ps(eta_ltd);
    __m256 v_w_min = _mm256_set1_ps(w_min);
    __m256 v_dcn_base = _mm256_set1_ps(100.0f);
    __m256 v_zero = _mm256_setzero_ps();

    size_t ltd_events = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t s = 0; s < steps; ++s) {
        for (size_t i = 0; i < num_synapses; i += 8) {
            __m256 v_pf = _mm256_loadu_ps(&pf_activity[i]);
            __m256 v_w = _mm256_loadu_ps(&pf_weights[i]);

            // Simple spike transmission: PF * W
            __m256 v_trans = _mm256_mul_ps(v_pf, v_w);

            // Heterosynaptic LTD update: W_new = max(w_min, W - eta_ltd * PF)
            __m256 v_dw = _mm256_mul_ps(v_eta_ltd, v_pf);
            __m256 v_w_new = _mm256_max_ps(v_w_min, _mm256_sub_ps(v_w, v_dw));
            _mm256_storeu_ps(&pf_weights[i], v_w_new);

            // DCN Disinhibition: DCN = max(0, DCN_base - 1.2 * Trans)
            __m256 v_dcn = _mm256_max_ps(v_zero, _mm256_sub_ps(v_dcn_base, _mm256_mul_ps(_mm256_set1_ps(1.2f), v_trans)));

            __m256 active_mask = _mm256_cmp_ps(v_dcn, _mm256_set1_ps(50.0f), _CMP_GT_OQ);
            int mask_bits = _mm256_movemask_ps(active_mask);
            ltd_events += __builtin_popcount(mask_bits);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration_sec = std::chrono::duration<double>(t_end - t_start).count();
    double total_evaluations = static_cast<double>(num_synapses) * steps;
    double millions_per_sec = (total_evaluations / duration_sec) / 1e6;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "WEFT AVX2 Cerebellar Purkinje-DCN Microbenchmark" << std::endl;
    std::cout << "  Parallel Fibers:" << num_synapses << std::endl;
    std::cout << "  Timesteps:      " << steps << std::endl;
    std::cout << "  Elapsed Time:   " << std::fixed << std::setprecision(4) << duration_sec << " s" << std::endl;
    std::cout << "  Throughput:     " << std::fixed << std::setprecision(2) << millions_per_sec << " Million synapse-steps/sec" << std::endl;
    std::cout << "  DCN Disinhibitions: " << ltd_events << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}

int main(int argc, char** argv) {
    size_t synapses = 131072; // 128k PF -> Purkinje synapses
    size_t steps = 1000;
    if (argc > 1) synapses = std::stoull(argv[1]);
    if (argc > 2) steps = std::stoull(argv[2]);

    run_cerebellar_avx2_bench(synapses, steps);
    return 0;
}
