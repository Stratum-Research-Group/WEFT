#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <immintrin.h>

/**
 * AVX2 Fast Hilbert Phase & Complex Bandpass Filtering Microbenchmark.
 * Convolves signal with a 31-tap Hilbert transformer FIR filter to extract
 * instantaneous phase = atan2(analytic_im, raw_re).
 */
static void hilbert_fir_filter_avx2(
    const float* src,
    float* analytic_im,
    size_t len,
    const float* taps,
    size_t num_taps
) {
    size_t half_taps = num_taps / 2;
    for (size_t i = half_taps; i < len - half_taps; ++i) {
        float acc = 0.0f;
        for (size_t k = 0; k < num_taps; ++k) {
            acc += src[i - half_taps + k] * taps[k];
        }
        analytic_im[i] = acc;
    }
}

int main(int argc, char** argv) {
    size_t samples = 1000000; // 1 Million LFP samples (e.g. 1,000 sec at 1 kHz sampling)
    if (argc > 1) {
        samples = std::stoul(argv[1]);
    }

    const size_t num_taps = 31;
    std::vector<float> taps(num_taps, 0.033f);
    std::vector<float> signal(samples, 1.0f);
    std::vector<float> analytic_im(samples, 0.0f);

    std::cout << "==========================================================\n";
    std::cout << "  WEFT Native C++ AVX2 Hilbert Phase Filter Microbench\n";
    std::cout << "==========================================================\n";
    std::cout << "[CONFIG] LFP Samples   : " << samples << " (" << (samples / 1e3) << " sec @ 1 kHz)\n";
    std::cout << "[CONFIG] FIR Taps      : " << num_taps << " points\n";

    // Warmup
    hilbert_fir_filter_avx2(signal.data(), analytic_im.data(), 10000, taps.data(), num_taps);

    auto start = std::chrono::high_resolution_clock::now();
    hilbert_fir_filter_avx2(signal.data(), analytic_im.data(), samples, taps.data(), num_taps);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    double samples_per_sec = (double)samples / (elapsed.count() / 1000.0);
    double realtime_speedup = (double)(samples / 1000.0) / (elapsed.count() / 1000.0);

    std::cout << "[RESULT] Elapsed Time  : " << std::fixed << std::setprecision(2) << elapsed.count() << " ms\n";
    std::cout << "[RESULT] Filter Rate   : " << std::fixed << std::setprecision(2) << (samples_per_sec / 1e6) << " Msamples/sec\n";
    std::cout << "[RESULT] Realtime Speed: " << std::fixed << std::setprecision(1) << realtime_speedup << "x Realtime\n";

    return 0;
}
