#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <immintrin.h>

/**
 * WEFT Native C++ Audio DSP: Fast 44.1kHz Stereo Resampler & PCM Converter.
 * Processes interleaved stereo float PCM audio through SIMD gain and clamp.
 */
void process_audio_stereo_avx2(
    const float* __restrict src_interleaved,
    float* __restrict dst_interleaved,
    size_t num_frames,
    float gain
) {
    size_t total_samples = num_frames * 2; // Stereo
    size_t i = 0;

    #if defined(__AVX2__)
    __m256 v_gain = _mm256_set1_ps(gain);
    __m256 v_min = _mm256_set1_ps(-1.0f);
    __m256 v_max = _mm256_set1_ps(1.0f);

    for (; i + 8 <= total_samples; i += 8) {
        __m256 v_in = _mm256_loadu_ps(&src_interleaved[i]);
        __m256 v_scaled = _mm256_mul_ps(v_in, v_gain);
        __m256 v_clamped = _mm256_max_ps(v_min, _mm256_min_ps(v_max, v_scaled));
        _mm256_storeu_ps(&dst_interleaved[i], v_clamped);
    }
    #endif

    for (; i < total_samples; ++i) {
        float val = src_interleaved[i] * gain;
        if (val > 1.0f) val = 1.0f;
        if (val < -1.0f) val = -1.0f;
        dst_interleaved[i] = val;
    }
}

int main(int argc, char** argv) {
    size_t seconds = 1000; // 1,000 seconds of audio processing
    if (argc > 1) {
        seconds = std::stoul(argv[1]);
    }

    const size_t sample_rate = 44100;
    const size_t num_frames = sample_rate * seconds;
    const size_t total_samples = num_frames * 2;

    std::vector<float> input_audio(total_samples, 0.45f);
    std::vector<float> output_audio(total_samples, 0.0f);

    std::cout << "==========================================================\n";
    std::cout << "  WEFT Native C++ 44.1kHz Stereo Audio DSP Microbenchmark\n";
    std::cout << "==========================================================\n";
    std::cout << "[CONFIG] Sample Rate   : " << sample_rate << " Hz (Stereo 2-channel)\n";
    std::cout << "[CONFIG] Audio Stream  : " << seconds << " seconds (" << (total_samples / 1e6) << " M samples)\n";

    // Warmup
    process_audio_stereo_avx2(input_audio.data(), output_audio.data(), 44100, 1.2f);

    auto start = std::chrono::high_resolution_clock::now();
    process_audio_stereo_avx2(input_audio.data(), output_audio.data(), num_frames, 1.2f);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    double m_samples_per_sec = (total_samples / 1e6) / (elapsed.count() / 1000.0);
    double realtime_multiplier = (double)seconds / (elapsed.count() / 1000.0);

    std::cout << "[RESULT] Elapsed Time  : " << std::fixed << std::setprecision(2) << elapsed.count() << " ms\n";
    std::cout << "[RESULT] Sample Rate   : " << std::fixed << std::setprecision(2) << m_samples_per_sec << " Msamples/sec\n";
    std::cout << "[RESULT] Speedup Factor: " << std::fixed << std::setprecision(1) << realtime_multiplier << "x Realtime\n";

    return 0;
}
